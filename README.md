# Russian News Сlustering and Headline Generation

## Введение
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

## Соревнование по кластеризации
Ссылка: https://competitions.codalab.org/competitions/28830
### Данные

Новостные документы берутся из [одноименного соревнования Телеграма](https://contest.com/docs/data_clustering2/ru). Поверх этого сделана попарная разметка документов в Толоке на предмет того, лежат ли документы в одном кластере.

[Инструкция по разметке](https://ilyagusev.github.io/purano/clustering_instruction.html)

Содержимое датасета: 
* ~15 тысяч размеченных пар новостей за 25 мая 2020, обучение и валидация: [ссылка](https://www.dropbox.com/s/8lu6dw8zcrn840j/ru_clustering_0525_urls.tsv)
* ~8,5 тысяч размеченных пар новостей за 27 мая 2020, публичный лидерборд: [ссылка](https://www.dropbox.com/s/3yh5ii20ijfbtb6/ru_clustering_0527_urls_final.tsv)
* ~8,5 тысяч размеченных пар новостей за 29 мая 2020, приватный лидерборд: [ссылка](https://www.dropbox.com/s/reria9xlfvj17a2/ru_clustering_0529_urls_final.tsv)

### Задача
Задача: кластеризация с эталонной разметкой или бинарная классификация

Метрики: F-мера для положительных пар.

В качестве бейзлайнов будут предлагались решения на основе именно кластеризации (полностью unsupervised, обучающая выборка только для подбора гиперпараметров). Однако, решения на основе бинарной классификации тоже принимались.

### Результаты

F-мера на положительных примерах.

| Login	             | Public LB | Private LB |
|:-------------------|:----------|:-----------|
| maelstorm          | 0,969     | 0,9604     |
| naergvae           | 0,967     | 0,9598     |
| g2tmn	             | 0,965     | 0,9573     |
| Kouki	             | 0,955     | 0,9548     |
| alexey.artsukevich | 0,958     | 0,9527     |
| smekur	           | 0,946     | 0,9387     |
| nikyudin	         | 0,938     | 0,9295     |
| landges	           | 0,916     | 0,9057     |
| kapant	           | 0,907     | 0,8985     |
| bond005	           | 0,902     | 0,8924     |
| anonym	           | 0,906     | 0,8910     |
| mashkka_t	         | 0,853     | 0,7149     |
| vatolinalex	       | 0,952     | 0,4760     |
| blanchefort	       | 0,941     |	          |
| imroggen	         | 0,903     |		        |
| Abiks	             | 0,894     |		        |
| dinabpr	           | 0,844     |		        |

## Соревнование по выбору заголовков

Ссылка: https://competitions.codalab.org/competitions/29479

### Данные
[Инструкция по разметке](https://ilyagusev.github.io/purano/selection_instruction.html)

* ~5 тысяч размеченных пар заголовков за 25 мая 2020, обучение и валидация: [ссылка](https://www.dropbox.com/s/jpcwryaeszqtrf9/titles_markup_0525_urls.tsv)
* ~3 тысячи размеченных пар заголовков за 27 мая 2020, публичный лидерборд: [ссылка](https://www.dropbox.com/s/jfa1b1xxw24znr9/titles_markup_0527_urls.tsv)
* ~3 тысячи размеченных пар заголовков за 29 мая 2020, приватный лидерборд: [ссылка](https://www.dropbox.com/s/qyegrt8oj2wn686/titles_markup_0529_urls.tsv)

### Задача
Задача: ранжирование заголовков

Метрики: точность на парах.

Безлайн: USE и Caboost в попарном режиме.

### Результаты

| Login	             | Public LB | Private LB |
|:-------------------|:----------|:-----------|
| sopilnyak          | 0,860     | 0,854      |
| landges            | 0,813	   | 0,820      |
| nikyudin           | 0,832	   | 0,816      |
| LOLKEK             | 0,808	   | 0,814      |
| maelstorm          | 0,818	   | 0,798      |
| a.korolev          | 0,658	   | 0,662      |

## Соревнование по генерации заголовков

Ссылка: https://competitions.codalab.org/competitions/29905

### Данные
* Тестовая выборка, 9-12 мая 2021, данные Телеграма: [ссылка](https://www.dropbox.com/s/9vlf6plbjqpbmea/headline_generation_answers.jsonl.tar.gz)

### Задача
Задача: генерация заголовков

Метрики: ROUGE, BLEU

Бейзлайны: Lead-1 и Encoder-Decoder на RuBERT

### Результаты

ROUGE = (ROUGE-1 + ROUGE-2 + ROUGE-L) / 3

| Login	  | ROUGE   | BLEU  |
|:--------|:--------|:------|
| LOLKEK  |	0,387	  | 0,695 |
| Rybolos |	0,292	  | 0,596 |

## Организаторы
* Илья Гусев, МФТИ
* Иван Смуров, ABBYY, МФТИ

[**Основная страница соревнования на CodaLab**](https://competitions.codalab.org/competitions/28830#learn_the_details)

[**Телеграм-чат соревнования**](https://t.me/dialogue_clustering)



