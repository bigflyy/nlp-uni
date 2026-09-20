"""
Программа ищет формулу из N слов со знаками "+" или "-", результат вычисления которой - вектор, близкий к целевому направлению, построенному по TARGET_WORDS (заданные целевые слова).
Вектор слова - его числовое представление в модели; формула - сумма или разность таких векторов.
Целевое направление - направление суммы нормализованных векторов TARGET_WORDS 
(иначе говоря, направление, имеющее максимальную среднюю косинусную близость со всеми словами из TARGET_WORDS. Для двух слов оно совпадает с биссектрисой угла между их векторами). 
Боковые отклонения (+r, -r) перпендикулярны ему, чтобы они содержали только движение в сторону, а не вдоль цели. Тогда +r и -r симметрично отклоняют два вектора и при сложении взаимно уничтожаются.
Успех - все цели вошли в топ результатов.

Как устроен поиск:
1. Из векторов TARGET_WORDS строим целевое направление.
2. Из векторов случайно выбранных существительных удаляем часть вдоль целевого направления, получая боковые отклонения r.
3. Для каждого бокового отклонения создаем два желаемых направления: 
    target_weight * target_direction + side_weight * perpendicular_vector 
    target_weight * target_direction - side_weight * perpendicular_vector 

    Если N нечетное, добавляем target_direction.
4. Для каждого направления находим ближайшее существительное и составляем формулу.
5. Вычисляем вектор формулы и проверяем, вошли ли все TARGET_WORDS в его топ результатов.
"""

from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors

# Целевые слова, которые должны одновременно попасть в топ ближайших результатов.
TARGET_WORDS = ["игрок", "комбинация"]
# Слова, которые нельзя использовать в формуле.
EXCLUDED_WORDS = ["дживать", "спалтраница", "ребеночекать"]
# Проверяем формулы с таким количеством слов. Минимум - 2.
N_WORDS_VALUES = (2, 3, 4, 5)
# Количество случайных попыток построить формулу.
ATTEMPTS = 256
# Начальное значение генератора случайных чисел для повторяемости результатов.
RANDOM_SEED = 43
# Границы доли целевого направления при построении желаемых направлений слов.
MIN_TARGET_WEIGHT = 0.25
MAX_TARGET_WEIGHT = 0.75
# Количество ближайших результатов, среди которых ищутся все целевые слова.
RESULT_COUNT = 10
# Максимальное количество успешных формул, показываемых в конце.
DISPLAY_COUNT = 32

MODEL_FILE = Path(__file__).with_name("cbow.txt")
NOUN_SUFFIX = "_NOUN"

# Преобразует слово в ключ модели, добавляя суффикс существительного.
def noun_key(word):
    return word.lower() + NOUN_SUFFIX

# Преобразует формулу в текстовый вид.
def formula_text(signs, word_keys):
    text = ""
    # Записываем слова и их знаки в одну читаемую строку.
    for index, (sign, word_key) in enumerate(zip(signs, word_keys)):
        word = word_key.removesuffix(NOUN_SUFFIX)
        if index == 0:
            text = word if sign == 1 else "- " + word
        else:
            text += (" + " if sign == 1 else " - ") + word
    return text

# Возвращает индексы элементов с наибольшей косинусной близостью.
def find_top_indices(scores, count):
    indices_from_worst_to_best = np.argsort(scores)
    indices_from_best_to_worst = np.flip(indices_from_worst_to_best)
    return indices_from_best_to_worst[:count]

# Сначала ставит формулы с лучшими позициями целевых слов, затем - с большей
# косинусной близостью к целевому направлению.
def result_sort_key(result):
    worst_target_position = result[0]
    target_similarity = result[1]
    return worst_target_position, -target_similarity

def run_search(model, n_words):
    if n_words < 2:
        raise ValueError("Количество слов должно быть не меньше 2")

    # Проверяем, что в модели есть все целевые слова
    target_noun_keys = [noun_key(word) for word in TARGET_WORDS]
    missing_target_keys = [word_key for word_key in target_noun_keys if word_key not in model]
    if missing_target_keys:
        raise ValueError("В модели нет слов: " + ", ".join(missing_target_keys))

    # Складываем нормализованные векторы целевых слов, получая целевое направление.
    # Для двух слов оно совпадает с биссектрисой угла между их векторами.
    # Нормализация суммы оставляет только направление, делая её длину равной 1.
    target_direction = sum(model.get_vector(word, norm=True) for word in target_noun_keys)
    target_direction /= np.linalg.norm(target_direction)

    # Оставляем существительные, исключая целевые и запрещённые слова.
    all_model_keys = np.array(model.index_to_key)
    noun_indices = np.array([index for index, word_key in enumerate(all_model_keys) if word_key.endswith(NOUN_SUFFIX)])
    noun_keys = all_model_keys[noun_indices]
    # Единичные векторы позволяют считать косинусную близость скалярным произведением.
    normalized_noun_vectors = model.get_normed_vectors()[noun_indices]

    forbidden_noun_keys = target_noun_keys + [noun_key(word) for word in EXCLUDED_WORDS]
    forbidden_noun_mask = np.isin(noun_keys, forbidden_noun_keys)
    allowed_noun_mask = np.logical_not(forbidden_noun_mask)
    allowed_noun_keys = noun_keys[allowed_noun_mask]
    allowed_noun_vectors = normalized_noun_vectors[allowed_noun_mask]

    rng = np.random.default_rng(RANDOM_SEED)
    candidate_formulas = []
    pair_count = n_words // 2

    # В каждой попытке строим один вариант формулы из N слов.
    for attempt in range(ATTEMPTS):
        # Случайно выбираем векторы существительных и удаляем из них часть,
        # направленную вдоль целевого направления. Остаются боковые отклонения r.
        perpendicular_vectors = allowed_noun_vectors[rng.choice(len(allowed_noun_vectors), pair_count, replace=False)]
        target_projections = perpendicular_vectors @ target_direction
        target_projections = np.expand_dims(target_projections, axis=1)
        perpendicular_vectors -= target_projections * target_direction
        # Нормализуем боковые отклонения, чтобы side_weight напрямую задавал
        # длину боковой составляющей.
        perpendicular_vectors /= np.linalg.norm(perpendicular_vectors, axis=1, keepdims=True,)

        target_weight = rng.uniform(MIN_TARGET_WEIGHT, MAX_TARGET_WEIGHT)
        side_weight = np.sqrt(1 - target_weight**2)
        # Желаемый вклад - идеальный вектор, который одно слагаемое формулы
        # должно добавить к общей сумме. Это ещё не векторы настоящих слов:
        # ниже для каждого такого вклада будет найдено ближайшее слово.
        desired_contribution_vectors = []
        # Из каждого r создаём пару желаемых вкладов с +r и -r.
        for perpendicular_vector in perpendicular_vectors:
            positive_side_direction = (target_weight * target_direction + side_weight * perpendicular_vector)
            negative_side_direction = (target_weight * target_direction - side_weight * perpendicular_vector)
            desired_contribution_vectors.append(positive_side_direction)
            desired_contribution_vectors.append(negative_side_direction)
        # Для нечётного N добавляем желаемое направление без бокового отклонения.
        if n_words % 2:
            desired_contribution_vectors.append(target_direction)
        desired_contribution_vectors = np.asarray(desired_contribution_vectors)

        # В первой попытке используем только "+", далее выбираем знаки случайно.
        if attempt == 0:
            signs = np.ones(n_words, dtype=int)
        else:
            signs = rng.choice((1, -1), n_words)
        # Столбец знаков применяет один знак ко всему желаемому направлению слова.
        signs_as_column = np.expand_dims(signs, axis=1)

        # Для знака "-" ищем слово около направления, противоположного желаемому.
        word_search_directions = desired_contribution_vectors * signs_as_column
        selected_noun_indices = np.argmax(word_search_directions @ allowed_noun_vectors.T, axis=1,)

        # Проверяем, что все выбранные слова разные.
        # Направления обрабатываются независимо и могут выбрать одно ближайшее слово.
        # Отбрасываем такой вариант, чтобы формула содержала N разных слов.
        unique_noun_indices = set(selected_noun_indices)
        if len(unique_noun_indices) != n_words:
            continue

        # Вычисляем вектор формулы как сумму найденных векторов с их знаками.
        formula_vector = (allowed_noun_vectors[selected_noun_indices] * signs_as_column).sum(axis=0)
        formula_length = np.linalg.norm(formula_vector)

        # Нормализуем вектор формулы: для косинусной близости важно направление.
        formula_vector /= formula_length
        formula_word_keys = tuple(allowed_noun_keys[selected_noun_indices])
        target_similarity = float(formula_vector @ target_direction)
        candidate_formulas.append((
            target_similarity,
            signs,
            formula_word_keys,
            formula_vector,
        ))

    noun_index_by_key = { word_key: index for index, word_key in enumerate(noun_keys) }
    successful_formulas = []

    # Для каждой формулы находим RESULT_COUNT ближайших существительных.
    for target_similarity, signs, formula_word_keys, formula_vector in candidate_formulas:
        noun_scores = normalized_noun_vectors @ formula_vector
        # Исключаем слова самой формулы из её ближайших результатов.
        for word_key in formula_word_keys:
            noun_scores[noun_index_by_key[word_key]] = -np.inf

        top_noun_indices = find_top_indices(noun_scores, RESULT_COUNT)
        top_noun_keys = list(noun_keys[top_noun_indices])
        target_positions = []
        # Для каждого целевого слова запоминаем его место среди ближайших результатов.
        for word_key in target_noun_keys:
            if word_key in top_noun_keys:
                position = top_noun_keys.index(word_key) + 1
            else:
                position = None
            target_positions.append(position)

        # Отбрасываем формулы, в топ которых вошли не все целевые слова.
        if any(position is None for position in target_positions):
            continue

        worst_target_position = max(target_positions)
        successful_formulas.append((
            worst_target_position,
            target_similarity,
            signs,
            formula_word_keys,
            target_positions,
        ))

    successful_formulas.sort(key=result_sort_key)

    result_lines = [
        f"MODEL_FILE = {MODEL_FILE.name}",
        f"TARGET_WORDS = {TARGET_WORDS}",
        f"EXCLUDED_WORDS = {EXCLUDED_WORDS}",
        f"N_WORDS = {n_words}",
        f"ATTEMPTS = {ATTEMPTS}",
        f"RANDOM_SEED = {RANDOM_SEED}",
        f"MIN_TARGET_WEIGHT = {MIN_TARGET_WEIGHT}",
        f"MAX_TARGET_WEIGHT = {MAX_TARGET_WEIGHT}",
        f"RESULT_COUNT = {RESULT_COUNT}",
        f"DISPLAY_COUNT = {DISPLAY_COUNT}",
        "",
        "Лучшие комбинации:",
    ]
    # Печатаем только успешные формулы со всеми целевыми словами.
    for _, target_similarity, signs, formula_word_keys, target_positions in (successful_formulas[:DISPLAY_COUNT]):
        target_places_text = ", ".join(f"{word}={position}" for word, position in zip(TARGET_WORDS, target_positions))
        result_lines.append(f"{formula_text(signs, formula_word_keys)}: " f"{target_places_text}, близость={target_similarity:.4f}")

    if not successful_formulas:
        result_lines.append("Подходящих комбинаций не найдено. Увеличьте ATTEMPTS или измените RANDOM_SEED.")

    result_text = "\n".join(result_lines)
    print("\n" + result_text)

    # Для каждого N сохраняем отдельный файл, чтобы результаты не перезаписывались.
    result_file = Path(__file__).with_name(f"result_n{n_words}.txt")
    result_file.write_text(result_text + "\n", encoding="utf-8")
    print(f"\nРезультаты сохранены в {result_file.name}")

def main():
    print("Загрузка модели...")
    model = KeyedVectors.load_word2vec_format(str(MODEL_FILE))

    # Последовательно запускаем поиск для каждого количества слов.
    for n_words in N_WORDS_VALUES:
        print(f"\nПоиск формул из {n_words} слов...")
        run_search(model, n_words)

if __name__ == "__main__":
    main()
