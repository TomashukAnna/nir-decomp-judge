# DevAI: пробный запуск и сравнение по задаче 39

## Версия и запуск

- MLflow run: `062509bc4fcb43be98430081deb9bfe1`.
- Коммит кода: `3c773111fdfd9f0ea5437afc004f2d3fc6ee9e34`; конфигурация: `configs/devai-smoke.yaml`; seed: `42`.
- Коммит DevAI: `43c2fa7b84ef0c1522e8250576e6798bace29dd0`; задача: `39_Drug_Response_Prediction_SVM_GDSC_ML`.
- Модель LiteLLM: `openai/iairlab/qwen3.8-27b`; max_tokens: `16384`.


## Сравнение

Источник человеческих меток: `agent-as-a-judge/benchmark/judgment/OpenHands/human_as_a_judge/39_Drug_Response_Prediction_SVM_GDSC_ML.json` из указанного коммита. Сопоставление по `requirement_id`. 
Источник LLM-меток: `nir-decomp-judge\results\devai\task39-local-004\verdicts`
Проверено однозначное соответствие всех семи ID. Поле `preferences` в сравнение не включено.

| ID | Требование (кратко) | Модель | Человек | Совпало |
|---|---|---|---|---|
| 0 | Загрузка датасета GDSC | PASS | FAIL | Нет |
| 1 | Отбор признаков | PASS | PASS | Да |
| 2 | Реализация SVM-регрессора | PASS | PASS | Да |
| 3 | Кросс-валидация | PASS | PASS | Да |
| 4 | Метрики в results/metrics/performance.txt | FAIL | PASS | Нет |
| 5 | Визуализация seaborn в results/figures/ | FAIL | FAIL | Да |
| 6 | PDF-отчёт с описанием работы и визуализацией | PASS | PASS | Да |


## Разбор требований: совпадений, расхождений и ограничений

- **ID 0: "The "GDSC" drug response dataset is loaded in `src/data_loader.py`."** 
Модель признала достаточными вызов `pd.read_csv` и наличие `gdsc_dataset.csv`. Человеческая метка — FAIL. Здесь нельзя точно утверждать, что модель не права, так как содержимое CSV не передавалось модели. Обоснования, почему этот критерий FAIL по мнению экспертов - нет.
- **ID 1: "Feature selection is performed to identify important features in `src/data_loader.py`**
Здесь справедливо PASS в обоих вариантах. LLM нашла функции в нужном файле по отбору признаков: performs feature selection using SelectKBest with f_regression and returns the selected feature names.
- **ID 2: "The "SVM regressor" is implemented in `src/model.py`**
Оба варианта опять справедливо PASS, так как SVM-regressor действительно есть.
- **ID 3: "Cross-validation is used to evaluate the model in `src/train.py`**
Тоже всё корректно, модель указывает: src/train.py evaluates the model using sklearn cross_val_score with 5-fold cross-validation."
- **ID 4: The performance results are saved as `results/metrics/performance.txt`** Модель поставила FAIL, поскольку в опубликованном дереве метрики лежат в `results/performance.txt`, а требование указывает `results/metrics/performance.txt`. В то же время `src/train.py` записывает метрики по требуемому пути. Человеческая метка — PASS; объяснения в разметке нет. Зафиксировано расхождение между артефактами и кодом, без изменения эталона.
- **ID 5: The regression results are visualized using "seaborn" and saved to `results/figures/`** обе оценки — FAIL. Модель указала, что результаты сохранены в неверную папку, а тажке, что построена гистограмма RMSE вместо визуализации связи признаков и отклика. В коде действительно используется `sns.histplot(rmse_scores, kde=True)`. Причина человеческого FAIL неизвестна, но логично, что она похожа.
- **ID 6: A report containing data preprocessing, model training, evaluation process, and the regression results visualization, is created and saved as `results/drug_response_prediction_report.pdf`** Модель поставила PASS по наличию PDF и содержимому соседнего Markdown-отчета. Содержимое PDF и изображения не передавалось, поэтому это ограниченная оценка по текстовым свидетельствам, а не проверка самого PDF.

## Сохранённые результаты

- [raw-response.txt](raw-response.txt) — исходный текст ответа, скопирован побайтно без исправлений.
- [verdicts.json](verdicts.json) — проверенный JSON с семью вердиктами и причинами модели.
- [comparison.csv](comparison.csv) — полные критерии, оценки, совпадения, причины модели и ID запуска.
- [run.json](run.json) и [effective-config.json](effective-config.json) — исходные сведения о запуске, скопированы без изменений.

