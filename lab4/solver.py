"""
Ищем промпт, после которого ruGPT-3 large сразу дописывает два заданных слова
в заданном порядке и точно в заданной форме.

Как работает программа:
1. Составляем список разрешенных слов из словаря модели: токены с пробелом впереди,
   которые являются целыми русскими словами от 3 букв и известны pymorphy3.
   Из списка убираем:
   - все формы обоих слов: для "спиной" - "спина", "спину", "спины" и другие;
   - слова с 3 общими буквами подряд с одним из слов: "испуг" для "вспугнуть";
   - слова с общим корнем по таблице корней из lab3: "скажите" для "приказу".
   Слова, которых нет в таблице корней, остаются.
2. Начинаем с промпта из 24 токенов: первый всегда "-", остальные - случайные
   разрешенные слова. Дефис нужен, чтобы текст промпта не начинался с пробела:
   его легко потерять при копировании, и тогда модель получит другие токены.
3. Считаем loss: сумму кросс-энтропий токенов пары, поставленных сразу после промпта.
   Чем вероятнее модель продолжит промпт именно этой парой, тем меньше loss.
4. На каждом шаге меняем одну позицию промпта, по очереди: 1, 2, ..., 23,
   потом снова с 1-й (на позиции 0 всегда дефис). По градиенту loss'а
   (формула Тейлора) выбираем 64 слова, которые должны сильнее всего снизить loss,
   и считаем для каждого настоящий loss. Оставляем лучший вариант; текущий промпт
   тоже среди вариантов, поэтому loss не растет.
5. После каждого шага превращаем промпт в текст, токенизируем его заново и запускаем
   жадный generate. Успех, если продолжение начинается с пары, а второе слово
   не оборвано.
6. Каждые 200 шагов начинаем заново со случайного промпта, всего до 2000 шагов.
   Печатаем найденный промпт вместе с продолжением модели или None.

Запуск: python solver.py вспугнуть спиной
С другим сидом случайного начального промпта: python solver.py вспугнуть спиной 6
"""
import gzip
import json
import re
import sys
from pathlib import Path

import pymorphy3
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Модель из задания: ruGPT-3 large от Сбера.
MODEL_NAME = 'ai-forever/rugpt3large_based_on_gpt2'
# Длина промпта в токенах вместе с первым токеном "-".
PROMPT_LENGTH = 24
# Первый токен каждого промпта, его поиск не меняет. Без него текст промпта начинался бы
# с пробела первого слова, а этот пробел легко потерять при копировании.
FIRST_TOKEN_TEXT = '-'
# Минимальная длина слова в промпте, в буквах. Короткие токены вроде "на", "ж"
# часто оказываются кусками слов, а не словами.
MIN_WORD_LETTERS = 3
# Сколько лучших по градиенту слов на каждом шаге проверяем настоящим прогоном модели.
CANDIDATE_COUNT = 64
# Шагов в одной попытке. Потом начинаем заново со случайного промпта.
STEPS_PER_ATTEMPT = 200
# Всего шагов на пару: 10 попыток по 200. Если промпт не найден, сдаемся.
MAX_STEPS = 2000
# Слово с тремя общими буквами подряд с целью считаем родственным: "напуган" для "вспугнуть".
SHARED_LETTERS = 3
# Путь к таблице корней. Это копия таблицы из lab3, лежит рядом с решателем.
ROOTS_PATH = Path(__file__).resolve().parent / 'resources' / 'roots.json.gz'
# Сколько токенов generate пишет после пары при проверке. По первому из них видно,
# не продолжается ли второе слово буквами: "кот" + "ёнок" дало бы "котёнок", не ту форму.
# Остальные показывают, что модель пишет дальше. Для самой проверки хватило бы 1.
EXTRA_TOKENS_AFTER_PAIR = 6
# Сид по умолчанию: задает случайный начальный промпт. Другой сид дает другой промпт.
SEED = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval().requires_grad_(False)
word_vectors = model.transformer.wte.weight
morphology = pymorphy3.MorphAnalyzer()
# Слово -> список его корневых семей; слова с ё записаны через е. Таблица приближенная.
roots_by_word = json.load(gzip.open(ROOTS_PATH, 'rt', encoding='utf-8'))['roots_by_word']


# Считаем для каждого промпта батча сумму кросс-энтропий токенов пары.
def target_loss(prompt_vectors, target_ids):
    batch_size = len(prompt_vectors)
    target_vectors = word_vectors[target_ids].repeat(batch_size, 1, 1)
    logits = model(inputs_embeds=torch.cat([prompt_vectors, target_vectors], dim=1)).logits
    target_logits = logits[:, PROMPT_LENGTH - 1:-1].transpose(1, 2)
    return torch.nn.functional.cross_entropy(
        target_logits, target_ids.repeat(batch_size, 1), reduction='none').sum(dim=1)


# Берем словарные слова без форм целей и без их кусков. Токен GPT-2 хранит пробел перед словом.
def allowed_tokens(first_word, second_word):
    targets = [word.lower().replace('ё', 'е') for word in (first_word, second_word)]
    # Все словоформы обеих целей по всем разборам pymorphy: "спиной", "спина", "спину" и так далее.
    banned_forms = {form.word.replace('ё', 'е') for target in targets
                    for parsed in morphology.parse(target) for form in parsed.lexeme}
    # Все куски целей по 3 буквы подряд: для "спиной" это "спи", "пин", "ино", "ной".
    banned_parts = {target[start:start + SHARED_LETTERS] for target in targets
                    for start in range(len(target) - SHARED_LETTERS + 1)}
    # Корни целей собираем по всем словоформам: в таблице может не быть именно этой формы.
    banned_roots = {root for form in banned_forms for root in roots_by_word.get(form, [])}
    allowed_ids = []
    # Перебираем весь словарь модели и оставляем только токены, которые можно ставить в промпт.
    for token_id in range(len(tokenizer)):
        token_text = tokenizer.decode([token_id])
        word = token_text[1:].replace('ё', 'е')
        # Токен годится, если это русское слово из строчных букв с пробелом перед ним,
        # не короче MIN_WORD_LETTERS букв, не форма цели, без куска цели из SHARED_LETTERS
        # букв, без общего корня с целью по таблице и известное словарю pymorphy.
        if (re.fullmatch(' [а-яё]+', token_text) and len(word) >= MIN_WORD_LETTERS
                and word not in banned_forms
                and not any(part in word for part in banned_parts)
                and banned_roots.isdisjoint(roots_by_word.get(word, []))
                and morphology.word_is_known(word)):
            allowed_ids.append(token_id)
    return torch.tensor(allowed_ids, device=device)


# Ищем промпт для пары слов; если не нашли, возвращаем None.
def solve(first_word, second_word, seed=SEED):
    target_text = f' {first_word} {second_word}'
    target_ids = tokenizer.encode(target_text, return_tensors='pt').to(device)
    allowed_ids = allowed_tokens(first_word, second_word)
    torch.manual_seed(seed)

    # Главный цикл поиска. На каждом шаге меняем одно слово промпта и сразу проверяем
    # промпт настоящим generate. Выходим, как только модель дописала пару.
    for step in range(MAX_STEPS):
        # В начале каждой попытки, раз в STEPS_PER_ATTEMPT шагов, берем новый случайный
        # промпт из разрешенных слов.
        if step % STEPS_PER_ATTEMPT == 0:
            prompt_ids = allowed_ids[torch.randint(len(allowed_ids), (1, PROMPT_LENGTH), device=device)]
            # На позицию 0 ставим FIRST_TOKEN_TEXT, дефис.
            prompt_ids[0, 0] = tokenizer.encode(FIRST_TOKEN_TEXT)[0]

        # Позиции меняем по очереди: 1, 2, ..., PROMPT_LENGTH - 1 (то есть 23), потом снова
        # с 1-й; на позиции 0 всегда дефис.
        position = 1 + step % (PROMPT_LENGTH - 1)
        prompt_vectors = word_vectors[prompt_ids].requires_grad_()
        loss = target_loss(prompt_vectors, target_ids).sum()
        gradient = torch.autograd.grad(loss, prompt_vectors)[0][0, position]
        # Приблизительно оцениваем, как замена слова на этой позиции изменит loss L.
        # Для одного числа формула Тейлора: L(c) ~ L(a) + L'(a) * (c - a).
        # Здесь a - вектор текущего слова, c - вектор кандидата на замену.
        # Поэтому складываем изменения по всем 1536 координатам вектора:
        # L(c) ~ L(a) + sum_i(g_i * (c_i - a_i)), где g_i = dL/da_i - градиент.
        # Получаем L(c) ~ L(a) - sum_i(g_i * a_i) + sum_i(g_i * c_i).
        # Первые два слагаемых одинаковы для всех кандидатов на этой позиции.
        # Меньшая сумма g_i * c_i предсказывает меньший loss после замены.
        # @ считает эту сумму сразу для всех разрешенных слов.
        scores = word_vectors[allowed_ids] @ gradient
        # Нулевой вариант остается текущим промптом, поэтому loss никогда не растет.
        # Остальные CANDIDATE_COUNT вариантов получают слова с самыми маленькими оценками:
        # largest=False.
        variants = prompt_ids.repeat(CANDIDATE_COUNT + 1, 1)
        variants[1:, position] = allowed_ids[scores.topk(CANDIDATE_COUNT, largest=False).indices]
        variant_losses = target_loss(word_vectors[variants], target_ids)
        prompt_ids = variants[variant_losses.argmin(dim=0, keepdim=True)]

        # Проверяем промпт как текст: превращаем в строку и токенизируем заново, как это
        # сделает пользователь. Генерируем пару и еще EXTRA_TOKENS_AFTER_PAIR токенов.
        # Жадная генерация без штрафа за повторы, как в README для generate из задания.
        prompt_text = tokenizer.decode(prompt_ids[0])
        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
        output_ids = model.generate(**inputs, max_new_tokens=target_ids.shape[1] + EXTRA_TOKENS_AFTER_PAIR,
                                    do_sample=False, num_beams=1, repetition_penalty=1.0,
                                    pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:])
        # Успех, если продолжение начинается с пары, а после второго слова стоит не буква:
        # пробел, знак препинания или перевод строки. Значит, слово не оборвано.
        if re.match(re.escape(target_text) + r'\W', generated_text):
            return prompt_text, generated_text
    return None


if __name__ == '__main__':
    # Третий аргумент необязательный: сид, без него берется SEED.
    seed_from_command_line = int(sys.argv[3]) if len(sys.argv) > 3 else SEED
    print(solve(sys.argv[1], sys.argv[2], seed_from_command_line))
