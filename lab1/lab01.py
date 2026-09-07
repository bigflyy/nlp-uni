from pathlib import Path
import sys

import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize


TEXT_FILE = Path(__file__).with_name("text.txt")
RESULT_FILE = Path(__file__).with_name("result.txt")
WORD_PARTS = {"NOUN", "ADJF"}

def words_agree(first, second):
    # Число и падеж должны быть известны и совпадать
    if None in (first.tag.number, second.tag.number, first.tag.case, second.tag.case):
        return False
    if first.tag.number != second.tag.number or first.tag.case != second.tag.case:
        return False

    # Во множественном числе форма не выражает род
    if first.tag.number == "plur":
        return True

    return first.tag.gender is not None and first.tag.gender == second.tag.gender


def find_pair(first_word, second_word, morph):
    # Ищем самый вероятный согласованный разбор
    first_parses = morph.parse(first_word)
    second_parses = morph.parse(second_word)

    if first_parses[0].tag.POS not in WORD_PARTS:
        return None
    if second_parses[0].tag.POS not in WORD_PARTS:
        return None

    best_pair = None
    best_score = -1

    for first in first_parses:
        if first.tag.POS not in WORD_PARTS:
            continue

        for second in second_parses:
            if second.tag.POS not in WORD_PARTS or not words_agree(first, second):
                continue

            score = first.score * second.score
            if score > best_score:
                best_score = score
                best_pair = first.normal_form, second.normal_form

    return best_pair


def main():
    # Настраиваем правильный вывод русского текста
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    morph = pymorphy3.MorphAnalyzer()
    text = TEXT_FILE.read_text(encoding="utf-8")
    result = []

    # Обрабатываем соседние слова внутри каждого предложения
    for sentence in sent_tokenize(text, language="russian"):
        tokens = word_tokenize(sentence, language="russian", preserve_line=True)

        for first_word, second_word in zip(tokens, tokens[1:]):
            pair = find_pair(first_word, second_word, morph)
            if pair is not None:
                result.append(" ".join(pair))

    # Сохраняем каждую пару на отдельной строке
    RESULT_FILE.write_text("\n".join(result), encoding="utf-8")


if __name__ == "__main__":
    main()
