"""
Ищем текст с [MASK], в котором оба заданных слова входят в топ-10 предсказаний BERT.

Как работает программа:
1. Приводим слова к нижнему регистру; требуем разбор pymorphy3
   и один токен BERT с тем же написанием.
2. Во всем тексте запрещаем их формы: для "дом" - "дом", "дома", "дому" и другие.
3. Находим подсказки по косинусной близости векторов BERT, исключая формы целей.
   Подставляем по 1, 2, 4 или 8 подсказок каждого слова в три предопределенные фразы с [MASK].
4. Токенизируем тексты и проверяем BERT пакетами: входят ли обе цели в топ-10?
5. При неудаче считаем loss: -log(p1)-log(p2), где p1 и p2 - вероятности
   двух заданных слов на месте [MASK]. Чем они выше, тем меньше loss.
   По градиенту loss'а по входным векторам предлагаем одиночные замены слов.
   Проверяем их BERT и продолжаем из двух лучших текстов, меняя подсказки и слова фразы.
6. Декодируем ответ и проверяем его отдельно стандартным вызовом BERT.
   Печатаем текст и топ-10; неудачу отмечаем явно.
Синонимы и однокоренные слова разрешены.
Результирующий текст содержит списки подсказок в кавычках;
"""

import argparse
import re
import time
from operator import itemgetter

import pymorphy3
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# Название модели BERT, которая предсказывает слово на месте [MASK].
MODEL_NAME = 'ai-forever/ruBert-large'
# Число вычислительных потоков PyTorch на процессоре.
CPU_THREADS = 4
# Максимальное число предложений, проверяемых одним вызовом BERT.
BATCH_SIZE = 32
# Число токенов выходного топа BERT, куда должны попасть обе цели.
TOP_WORDS = 10
# Число ближайших к цели токенов, которые просматриваем при подборе подсказок.
NEIGHBORS_TO_CHECK = 256
# Число ближайших подсказок каждой цели в исходных текстах: 1, 2, 4 или 8.
CLUE_COUNTS = (1, 2, 4, 8)
# Мягкий лимит секунд только на цикл направленных замен; весь запуск может быть дольше.
SEARCH_SECONDS = 15
# Число перспективных слов, проверяемых BERT для каждой изменяемой позиции.
REPLACEMENTS_PER_POSITION = 4
# Число разных предложений, из которых одновременно продолжаем направленный поиск.
BEAM_SIZE = 2

# Три фразы с перечислением подсказок вокруг одной маски.
TEMPLATES = (
    'Это не {left} и {right}, а "[MASK]".',
    'Он записал {left}, "[MASK]" и {right}.',
    'Здесь написано {left}, "[MASK]", {right}.',
)


# Приводим слово к нижнему регистру и заменяем ё на е только для проверки запретов.
def normalize_word(word):
    return word.lower().replace('ё', 'е')


# Заключаем каждую подсказку в кавычки и соединяем подсказки через запятую.
def quoted_words(words):
    quoted = []
    for word in words:
        quoted.append('"' + word + '"')
    return ', '.join(quoted)


# Ищем контекст заменами любых целых слов, сохраняя пунктуацию и запрет словоформ.
class Solver:
    # Загружаем BERT, токенизатор и разборщик слов, готовим таблицы для поиска подсказок.
    def __init__(self):
        torch.set_num_threads(CPU_THREADS)
        # Устройство для модели и тензоров: процессор или доступная видеокарта.
        self.device = 'cpu'
        # При наличии CUDA выполняем вычисления на видеокарте.
        if torch.cuda.is_available():
            # Выбранная видеокарта для последующих вычислений.
            self.device = 'cuda'
        # Токенизатор модели с сохранением букв ё и й.
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=True, strip_accents=False
        )
        # Предобученный BERT для заполнения маски, загруженный из локального кэша.
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, local_files_only=True)
        self.model.to(self.device)
        # Фиксируем веса модели для поиска входного текста.
        self.model.requires_grad_(False)
        # Включаем ускоренные CUDA-умножения; по умолчанию этот режим выключен.
        torch.backends.cuda.matmul.allow_tf32 = True
        # Матрица входных векторов всех токенов словаря BERT.
        self.token_embeddings = self.model.get_input_embeddings().weight
        # Те же векторы единичной длины для поиска по косинусной близости.
        self.normalized_token_embeddings = torch.nn.functional.normalize(self.token_embeddings)
        # Написания токенов в порядке их ID в словаре BERT.
        self.vocabulary_tokens = self.tokenizer.convert_ids_to_tokens(range(len(self.tokenizer)))
        # Морфологический словарь для проверки слов и получения словоформ.
        self.morphology_analyzer = pymorphy3.MorphAnalyzer()
        # True: запись словаря BERT строчная и известна pymorphy3.
        self.standalone_word_token_flags = []
        # Отмечаем строчные записи словаря BERT, известные pymorphy3.
        for word in self.vocabulary_tokens:
            word_is_known = self.morphology_analyzer.word_is_known(word)
            is_lowercase_dictionary_word = word_is_known and word == word.lower()
            self.standalone_word_token_flags.append(is_lowercase_dictionary_word)
        # Те же признаки как тензор для отбора подсказок и позиций замены.
        self.standalone_word_token_mask = torch.tensor(
            self.standalone_word_token_flags, device=self.device
        )

    # Проверяем точное совпадение с одной из словарных форм целей, игнорируя регистр и ё/е.
    def is_forbidden_word(self, word):
        return normalize_word(word) in self.forbidden_wordforms

    # Проверяем отсутствие словоформ целей во всем тексте и наличие одной маски.
    def contains_one_mask_and_no_forbidden_words(self, sentence):
        # Проверяем все слова текста, включая постоянные слова шаблона.
        for word in re.findall('[А-Яа-яЁё]+', sentence):
            # Отклоняем текст при первой словоформе любой из двух целей.
            if self.is_forbidden_word(word):
                return False
        return sentence.count('[MASK]') == 1

    # Берем ближайшие словарные токены, исключая формы целей.
    def select_clues(self, target_token_id):
        # Считаем близость к цели и исключаем неподходящие записи словаря BERT.
        similarities = (
            self.normalized_token_embeddings @ self.normalized_token_embeddings[target_token_id]
        )
        similarities = similarities.masked_fill(~self.standalone_word_token_mask, -torch.inf)
        clue_words = []
        # Просматриваем ближайшие слова в порядке убывания косинусной близости.
        for token_id in similarities.topk(NEIGHBORS_TO_CHECK).indices.tolist():
            word = self.vocabulary_tokens[token_id]
            # Подсказка не должна быть словоформой ни одной цели.
            if not self.is_forbidden_word(word):
                clue_words.append(word)
            # В самом длинном исходном списке потребуется не больше подсказок.
            if len(clue_words) >= max(CLUE_COUNTS):
                break
        return clue_words

    # Подставляем списки ближайших подсказок в шаблоны и отбираем допустимые тексты.
    def initial_sentences(self, first_clues, second_clues):
        sentences = []
        # Строим тексты с разным числом подсказок для каждого целевого слова.
        for count in CLUE_COUNTS:
            left = quoted_words(first_clues[:count])
            right = quoted_words(second_clues[:count])
            sentence_count_before_templates = len(sentences)
            # Проверяем три фразы с одинаковыми списками подсказок.
            for template in TEMPLATES:
                sentence = template.format(left=left, right=right)
                # Сохраняем только текст с одной маской и без запрещенных форм.
                if self.contains_one_mask_and_no_forbidden_words(sentence):
                    sentences.append(sentence)
            # Например, цели "и" и "здесь" запрещают постоянные слова всех трех фраз.
            if len(sentences) == sentence_count_before_templates:
                sentence = '{left}, "[MASK]", {right}.'.format(left=left, right=right)
                # Резервное перечисление тоже должно соблюдать запрет словоформ.
                if self.contains_one_mask_and_no_forbidden_words(sentence):
                    sentences.append(sentence)
        return sorted(dict.fromkeys(sentences), key=len)

    # Выдаем результаты по одному; после успеха вызывающий код может не считать следующие пакеты.
    @torch.inference_mode()
    def evaluate(self, token_sequences):
        # Обрабатываем кандидатов пакетами, не превышая BATCH_SIZE.
        for start in range(0, len(token_sequences), BATCH_SIZE):
            batch = token_sequences[start:start + BATCH_SIZE]
            # Выравниваем длины и создаем attention_mask, чтобы BERT игнорировал паддинг.
            encoded_inputs = self.tokenizer.pad(
                {'input_ids': batch}, return_tensors='pt'
            ).to(self.device)
            sequence_indices = torch.arange(len(batch), device=self.device)
            mask_positions = (
                (encoded_inputs.input_ids == self.tokenizer.mask_token_id).int().argmax(1)
            )
            # Передаем input_ids и attention_mask; считаем выходные логиты только для масок.
            hidden = self.model.bert(**encoded_inputs).last_hidden_state
            logits = self.model.cls(hidden[sequence_indices, mask_positions])
            top_predictions = logits.topk(TOP_WORDS)
            lower_target_logits = logits[:, self.target_token_ids].min(-1).values
            tenth_largest_logits = top_predictions.values[:, -1]
            # Положительная разница означает, что оба логита выше границы десятки.
            margins = lower_target_logits - tenth_largest_logits
            # Возвращаем для каждого текста его ID, разницу логитов и попадание целей в топ-10.
            for input_token_ids, predicted_token_ids, margin in zip(
                batch, top_predictions.indices.tolist(), margins.tolist()
            ):
                yield dict(input_token_ids=input_token_ids, margin=margin,
                           success=self.target_token_ids[0] in predicted_token_ids
                           and self.target_token_ids[1] in predicted_token_ids)

    # Считаем окончательную десятку одиночным BERT на готовом тексте.
    @torch.inference_mode()
    def finalize_result(self, result, method):
        # Восстанавливаем текст, удаляем CLS/SEP и приводим пробелы к печатному виду.
        sentence = self.tokenizer.decode(result['input_token_ids'])
        sentence = sentence.replace('[CLS]', '').replace('[SEP]', '').strip()
        sentence = sentence[:1].upper() + sentence[1:]
        sentence = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', sentence)
        sentence = re.sub(r'\s+([,.])', r'\1', sentence)
        # Проверяем именно готовый текст отдельным стандартным вызовом модели.
        encoded_inputs = self.tokenizer(sentence, return_tensors='pt').to(self.device)
        mask_position = encoded_inputs.input_ids[0].tolist().index(self.tokenizer.mask_token_id)
        logits = self.model(**encoded_inputs).logits[0, mask_position]
        top_predictions = logits.topk(TOP_WORDS)
        predicted_token_ids = top_predictions.indices.tolist()
        return dict(sentence=sentence, method=method,
                    success=self.target_token_ids[0] in predicted_token_ids
                    and self.target_token_ids[1] in predicted_token_ids,
                    margin=float(logits[self.target_token_ids].min() - top_predictions.values[-1]),
                    top10=self.tokenizer.convert_ids_to_tokens(predicted_token_ids))

    # Собираем номера словарных токенов, не являющихся формами текущих целей.
    def replacement_ids(self):
        replacement_token_ids = []
        # Собираем словарные токены, пригодные для вставки в текст.
        for token_id, word_is_known in enumerate(self.standalone_word_token_flags):
            # Исключаем словоформы обеих целей из всех замен.
            if word_is_known and not self.is_forbidden_word(self.vocabulary_tokens[token_id]):
                replacement_token_ids.append(token_id)
        return torch.tensor(replacement_token_ids, device=self.device)

    # Предлагаем несколько лучших замен каждого целого слова по градиенту.
    def propose_replacements(self, token_sequences, replacement_ids, seen_sequences):
        # Выравниваем входы и разрешаем вычислять производные по их векторам.
        encoded_inputs = self.tokenizer.pad(
            {'input_ids': token_sequences}, return_tensors='pt'
        ).to(self.device)
        input_vectors = self.token_embeddings[encoded_inputs.input_ids].detach().requires_grad_()
        # Получаем логиты маски с сохранением связей для вычисления градиента.
        hidden = self.model.bert(
            inputs_embeds=input_vectors, attention_mask=encoded_inputs.attention_mask
        ).last_hidden_state
        mask_positions = (encoded_inputs.input_ids == self.tokenizer.mask_token_id).int().argmax(1)
        logits = self.model.cls(
            hidden[torch.arange(len(token_sequences), device=self.device), mask_positions]
        )
        # Штраф -log(p1)-log(p2) уменьшается при росте вероятностей обеих целей.
        loss = -logits.log_softmax(-1)[:, self.target_token_ids].sum()
        gradients = torch.autograd.grad(loss, input_vectors)[0]
        # Находим заменяемые позиции - пары индексов (номер последовательности, позиция словарного токена).
        editable_positions = self.standalone_word_token_mask[encoded_inputs.input_ids].nonzero().T
        selected_gradients = gradients[editable_positions[0], editable_positions[1]]
        # Приблизительно оцениваем, как замена слова изменит штраф L.
        # Для одного числа формула Тейлора: L(c) ~ L(a) + L'(a) * (c - a).
        # Здесь a - вектор текущего слова, c - вектор кандидата на замену.
        # Поэтому складываем изменения по всем координатам вектора:
        # L(c) ~ L(a) + sum_i(g_i * (c_i - a_i)), где g_i = dL/da_i.
        # Получаем L(c) ~ L(a) - sum_i(g_i * a_i) + sum_i(g_i * c_i).
        # Первые два слагаемых одинаковы для всех кандидатов в одной позиции.
        # Меньшая сумма g_i * c_i предсказывает меньший штраф после замены.
        replacement_scores = -selected_gradients @ self.token_embeddings[replacement_ids].T
        suggested_token_ids = replacement_ids[
            replacement_scores.topk(REPLACEMENTS_PER_POSITION).indices
        ].tolist()
        proposals = []
        # Для каждой заменяемой позиции создаем отдельные варианты текста.
        for (sequence_index, word_position), suggested_ids in zip(
            editable_positions.T.tolist(), suggested_token_ids
        ):
            # В каждом варианте заменяем ровно один токен выбранным кандидатом.
            for new_token_id in suggested_ids:
                input_token_ids = token_sequences[sequence_index].copy()
                input_token_ids[word_position] = new_token_id
                # Убираем повторы всего входа, а не слова в разных позициях.
                key = tuple(input_token_ids)
                # Не добавляем уже проверенный или уже поставленный в очередь вход.
                if key not in seen_sequences:
                    proposals.append(input_token_ids)
                    seen_sequences.add(key)
        return proposals

    # Готовим запреты, проверяем шаблоны и при необходимости ищем одиночные замены.
    def solve(self, first_word, second_word):
        first_word = first_word.lower()
        second_word = second_word.lower()
        # ID двух введенных слов, которые должны войти в топ-10.
        self.target_token_ids = []
        # Все запрещенные словарные формы обеих целей в нормализованном написании.
        self.forbidden_wordforms = set()
        # Проверяем оба входных слова и собираем запреты до начала поиска.
        for word in (first_word, second_word):
            word_token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            # Для предсказания одной маской слово должно быть одним токеном с тем же написанием.
            if len(word_token_ids) != 1 or self.vocabulary_tokens[word_token_ids[0]] != word:
                raise ValueError('Нужен один токен для слова "' + word + '".')
            has_dictionary_parse = False
            # Учитываем все словарные значения слова, а не только первый разбор.
            for parsed in self.morphology_analyzer.parse(word):
                # Догадки анализатора не подтверждают словарные формы.
                if parsed.is_known:
                    has_dictionary_parse = True
                    # Запрещаем каждую форму подтвержденного словарного разбора.
                    for form in parsed.lexeme:
                        self.forbidden_wordforms.add(normalize_word(form.word))
            # Неизвестное словарю слово не позволяет надежно собрать запрет форм.
            if not has_dictionary_parse:
                raise ValueError('Нет словарного разбора для слова "' + word + '".')
            self.target_token_ids.append(word_token_ids[0])

        # Подбираем исходные подсказки и строим допустимые стартовые тексты.
        first_clues = self.select_clues(self.target_token_ids[0])
        second_clues = self.select_clues(self.target_token_ids[1])
        # Для начального текста нужны подсказки обеих целей.
        if not first_clues or not second_clues:
            raise ValueError('Не удалось подобрать словарные подсказки.')
        sentences = self.initial_sentences(first_clues, second_clues)
        # Завершаем с объяснением, если ни один исходный текст не прошел запреты.
        if not sentences:
            raise ValueError('Нет допустимого шаблона.')
        token_sequences = self.tokenizer(sentences).input_ids
        evaluated = []
        # Сначала пробуем решить задачу исходными текстами, без замен.
        for result in self.evaluate(token_sequences):
            # Кандидата с обеими целями в топ-10 передаем окончательной проверке.
            if result['success']:
                return self.finalize_result(result, 'templates')
            evaluated.append(result)
        # Если старт не помог, выбираем допустимые замены и две лучшие ветки поиска.
        replacement_ids = self.replacement_ids()
        current_results = sorted(evaluated, key=itemgetter('margin'), reverse=True)[:BEAM_SIZE]
        best_result = current_results[0]
        seen_sequences = set()
        # Запоминаем стартовые входы, чтобы не предлагать их повторно.
        for input_token_ids in token_sequences:
            seen_sequences.add(tuple(input_token_ids))
        started = time.perf_counter()
        # Начинаем следующий шаг, только пока не истек мягкий лимит времени.
        while time.perf_counter() - started < SEARCH_SECONDS:
            token_sequences = []
            # Передаем градиентному поиску текущие варианты текста.
            for result in current_results:
                token_sequences.append(result['input_token_ids'])
            proposals = self.propose_replacements(token_sequences, replacement_ids, seen_sequences)
            # Останавливаемся, если новых замен больше нет.
            if not proposals:
                break
            evaluated = []
            # Проверяем предложенные замены настоящими предсказаниями BERT.
            for result in self.evaluate(proposals):
                # Сохраняем лучший найденный ответ независимо от текущих веток.
                if result['margin'] > best_result['margin']:
                    best_result = result
                # Завершаем поиск, когда обе цели вошли в топ-10 кандидата.
                if result['success']:
                    return self.finalize_result(result, 'gradient')
                evaluated.append(result)
            # Продолжаем из двух лучших кандидатов этого шага.
            current_results = sorted(evaluated, key=itemgetter('margin'), reverse=True)[:BEAM_SIZE]
        return self.finalize_result(best_result, 'gradient')


# Читаем два слова из командной строки, запускаем поиск и печатаем предложение с топом BERT.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first_word')
    parser.add_argument('second_word')
    args = parser.parse_args()
    solver = Solver()
    # Запускаем поиск и превращаем неподдерживаемый ввод в понятную ошибку CLI.
    try:
        result = solver.solve(args.first_word, args.second_word)
    # Сообщаем причину отказа без Python traceback.
    except ValueError as error:
        parser.exit(2, str(error) + '\n')
    print(result['sentence'])
    print('Top-10:', ', '.join(result['top10']))
    # Отличаем найденное решение от лучшего неуспешного приближения.
    if not result['success']:
        print('Поиск завершён без решения.')
        raise SystemExit(1)

if __name__ == '__main__':
    main()
