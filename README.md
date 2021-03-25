# Russian News Сlustering and Headline Generation

## Описание задачи
Цель соревнования – собрать и сравнить подходы к кластеризации и выбору наилучшего заголовка для получившихся кластеров. Кластеризация новостей выглядит достаточно сложной задачей для современных моделей, и из-за этого является хорошим бенчмарком. Кроме того, кластеризация текстов как задача достаточно часто встречается в индустрии. Выбор или генерация лучшего заголовка – логичное её продолжение.

## Таймлайн соревнования
* **08.02.2021** -- запуск [дорожки по кластеризации](https://competitions.codalab.org/competitions/28830)
* **22.02.2021** -- — запуск [дорожки по выбору заголовков](https://competitions.codalab.org/competitions/29479)
* **01.03.2021**  -- запуск [дорожки по генерации заголовков](https://competitions.codalab.org/competitions/29905)
* **21.03.2021** -- **дедлайн по всем дорожкам.**
* **Вы находитесь здесь**
* **28.03.2021** -- дедлайн по подачи статей по результатам соревнования.

## Бейзлайны

https://colab.research.google.com/drive/1bam5oFul9Gzj9rryq8_7M1-EGJxO3K-G#scrollTo=GgLzfNigT-er

## [Соревнование по кластеризации](https://competitions.codalab.org/competitions/28830)
### Данные

Новостные документы берутся из [одноименного соревнования Телеграма](https://contest.com/docs/data_clustering2/ru). Поверх этого сделана попарная разметка документов в Толоке на предмет того, лежат ли документы в одном кластере.

[Инструкция по разметке](https://ilyagusev.github.io/purano/clustering_instruction.html)

Содержимое датасета: 
* (~15 тысяч размеченных пар новостей за 25 мая 2020, обучение и валидация)[https://www.dropbox.com/s/8lu6dw8zcrn840j/ru_clustering_0525_urls.tsv]
* (~8,5 тысяч размеченных пар новостей за 27 мая 2020, публичный лидерборд)[https://www.dropbox.com/s/3yh5ii20ijfbtb6/ru_clustering_0527_urls_final.tsv]
* (~8,5 тысяч размеченных пар новостей за 29 мая 2020, приватный лидерборд)[https://www.dropbox.com/s/reria9xlfvj17a2/ru_clustering_0529_urls_final.tsv]

### Задача
Задача: кластеризация с эталонной разметкой или бинарная классификация

Метрики: precision и recall для OK.

В качестве бейзлайнов будут предлагались решения на основе именно кластеризации (полностью unsupervised, обучающая выборка только для подбора гиперпараметров). Однако, решения на основе бинарной классификации тоже принимались.

## [Соревнование по выбору заголовков](https://competitions.codalab.org/competitions/29479)

### Данные
[Инструкция по разметке](https://ilyagusev.github.io/purano/selection_instruction.html)

* (~5 тысяч размеченных пар заголовков за 25 мая 2020, обучение и валидация)[https://www.dropbox.com/s/jpcwryaeszqtrf9/titles_markup_0525_urls.tsv]
* (~3 тысячи размеченных пар заголовков за 27 мая 2020, публичный лидерборд)[https://www.dropbox.com/s/jfa1b1xxw24znr9/titles_markup_0527_urls.tsv]
* (~3 тысячи размеченных пар заголовков за 27 мая 2020, приватный лидерборд)[https://www.dropbox.com/s/qyegrt8oj2wn686/titles_markup_0529_urls.tsv]

### Задача
Задача: ранжирование заголовков

Метрики: точность на парах.

Безлайн: USE и Caboost в попарном режиме.

## [Соревнование по генерации заголовков](https://competitions.codalab.org/competitions/29905)

### Данные
* (Тестовая выборка, 9-12 мая 2021, данные Телеграма)[https://www.dropbox.com/s/9vlf6plbjqpbmea/headline_generation_answers.jsonl.tar.gz]

### Задача
Задача: генерация заголовков

Метрики: ROUGE, BLEU

Бейзлайны: Lead-1 и Encoder-Decoder на RuBERT

## Организаторы
* Илья Гусев, МФТИ
* Иван Смуров, ABBYY, МФТИ

[**Страница соревнования на CodaLab**](https://competitions.codalab.org/competitions/28830#learn_the_details)

[**Телеграм-чат соревнования**](https://t.me/dialogue_clustering)



