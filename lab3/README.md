# Лабораторная 3

[solver_concise.py](solver_concise.py) подбирает текст с пропуском [MASK],
чтобы оба введенных слова вошли в топ-10 предсказаний BERT для этого пропуска.

## Установка и запуск

Проверено на Python 3.12. Команды для запуска:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
hf download ai-forever/ruBert-large
python solver_concise.py батарея кирпич
```

Модель загружается один раз в кэш Hugging Face. При доступной CUDA
программа использует видеокарту, иначе процессор.

## Результаты
[Все 200 пар и их предсказания](RESULTS.md)