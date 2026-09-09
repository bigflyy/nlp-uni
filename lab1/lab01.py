from pathlib import Path
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

# Файл с исходным текстом
TEXT_FILE = Path(__file__).with_name("text.txt")
# Файл для сохранения результатов
RESULT_FILE = Path(__file__).with_name("result.txt")

# Части слов, которые нас интересуют: существительные (NOUN) и полные прилагательные (ADJF)
WORD_PARTS = {"NOUN", "ADJF"}

# Функция для проверки согласованности двух слов по числу, падежу и роду
def words_agree(first, second):
    # Число, падеж и род должны быть известны 
    if None in (first.tag.number, second.tag.number, first.tag.case, second.tag.case, first.tag.gender, second.tag.gender):
        return False
    # Число, падеж и род должны совпадать
    if first.tag.number != second.tag.number or first.tag.case != second.tag.case or first.tag.gender != second.tag.gender:
        return False
    return True

# Функция для поиска лучшей пары согласованных слов среди всех возможных разборов двух слов
def find_pair(first_word, second_word, morph, min_score = 0.01):
    # Получаем разборы слов
    first_parses = morph.parse(first_word)
    second_parses = morph.parse(second_word)
    
    best_pair = None
    best_score = -1

    # Перебираем все варианты разборов для первого и второго слова и ищем лучшую пару (согласованную и с наибольшим score)
    for first in first_parses:
        # Берём только существительные и полные прилагательные 
        # Не берём местоимения в качестве прилагательных 
        # Не берём маловероятные варианты (score < min_score) (Например, местоимения, которые pymorphy определяет как существительные)
        if first.tag.POS not in WORD_PARTS or "Apro" in first.tag or first.score < min_score:
            continue
        for second in second_parses:
            # Берём только существительные и полные прилагательные 
            # Не берём местоимения в качестве прилагательных 
            # Не берём маловероятные варианты (score < min_score)
            if second.tag.POS not in WORD_PARTS or "Apro" in second.tag or second.score < min_score:
                continue
            # Проверяем согласованность по числу, падежу и роду
            if not words_agree(first, second):
                continue
            # Обновляем лучшую пару, если score больше
            score = first.score * second.score
            if score > best_score:
                best_score = score
                best_pair = first.normal_form, second.normal_form

    return best_pair


def main():
    morph = pymorphy3.MorphAnalyzer()
    text = TEXT_FILE.read_text(encoding="utf-8")
    result = []

    # Обрабатываем соседние слова внутри каждого предложения
    for sentence in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(sentence, language="russian", preserve_line=True)
        for first_word, second_word in zip(tokens, tokens[1:]):
            pair = find_pair(first_word, second_word, morph)
            if pair is not None:
                line = f"{first_word} {second_word} -> {' '.join(pair)}"
                result.append(" ".join(pair))
                print(line)

    # Сохраняем результат в файл
    RESULT_FILE.write_text("\n".join(result), encoding="utf-8")

if __name__ == "__main__":
    main()
