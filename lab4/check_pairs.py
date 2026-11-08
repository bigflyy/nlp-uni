"""
Проверяем solver.py на паре по заданию и на pairs.txt.

Каждый результат сохраняем сразу, поэтому прогон можно прервать:
при следующем запуске готовые пары пропускаются.
"""
import json
import statistics
import time
from pathlib import Path

from solver import solve

LAB_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_PATH = LAB_DIRECTORY / 'results_pairs.json'

# Первая пара - пара по заданию, остальные 30 читаем из файла.
pairs = [['вспугнуть', 'спиной']]
# Читаем файл по строкам и собираем пары слов.
for line in (LAB_DIRECTORY / 'pairs.txt').read_text(encoding='utf-8').splitlines():
    # Берем только строки из двух слов; строки с "#" в начале - комментарии.
    if len(line.split()) == 2 and not line.startswith('#'):
        pairs.append(line.split())

# Если результаты уже сохранены прошлым запуском, продолжаем с них, иначе начинаем с пустого списка.
rows = json.loads(OUTPUT_PATH.read_text(encoding='utf-8')) if OUTPUT_PATH.exists() else []
finished_pairs = [row['pair'] for row in rows]

# Решаем пары по очереди, замеряем время и сразу сохраняем результат в файл.
for first_word, second_word in pairs:
    # Пару, которая уже есть в файле, второй раз не решаем.
    if [first_word, second_word] in finished_pairs:
        continue
    start_time = time.perf_counter()
    result = solve(first_word, second_word)
    rows.append(dict(pair=[first_word, second_word], seconds=round(time.perf_counter() - start_time, 1),
                     prompt=result and result[0], generated=result and result[1]))
    print(rows[-1], flush=True)
    OUTPUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')

seconds = [row['seconds'] for row in rows]
print('Решено:', sum(row['prompt'] is not None for row in rows), 'из', len(rows))
print('Секунд на пару, медиана:', statistics.median(seconds), 'максимум:', max(seconds))
