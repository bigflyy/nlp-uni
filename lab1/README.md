# Лабораторная работа № 1

Необходимые библиотеки:
python -m pip install nltk pymorphy3 pymystem3

`pip install nltk` устанавливает только библиотеку, но не языковые данные.
Поэтому для разделения русского текста на предложения отдельно необходимо загрузить `punkt_tab`:
python -m nltk.downloader punkt_tab


Запуск основной версии NLTK + PyMorphy3:
python lab1/lab01.py

Читает исходный текст из `lab1/text.txt`.