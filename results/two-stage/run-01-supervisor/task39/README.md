# Задача 39 — запуск преподавателя

[Все три запуска](../../README.md) · [Вторая задача этого запуска](../task26/README.md)

Дата: 18 ноября 2026. Сравнение ниже относится только к этому запуску браузерного судьи.

## Описание задачи

Система прогнозирования ответа на лекарственные препараты по данным GDSC с помощью SVM-регрессора. Нужно загрузить данные, отобрать признаки, обучить модель и оценить её кросс-валидацией. Метрики сохраняются в results/metrics/performance.txt, визуализация seaborn — в results/figures/, PDF-отчёт — в results/drug_response_prediction_report.pdf. Отчёт должен объяснять влияние отбора признаков и показывать связь признаков с прогнозом. Система должна допускать добавление наборов данных и признаков. Источник: задача 39 DevAI, опубликованное приложение OpenHands.

### Исходное ТЗ

> Develop a system to predict drug response using the GDSC dataset with a Support Vector Machine (SVM) regressor. Load the dataset and perform feature selection to identify key features in `src/data_loader.py`. Implement the SVM regressor in `src/model.py`. Use cross-validation to evaluate the model's performance in `src/train.py`.  Save the performance results to `results/metrics/performance.txt`. Visualize the regression results using seaborn and save it under `results/figures/`. Next, create a report including the data preprocessing, model training, evaluation process, and the visualization. Save the report as `results/drug_response_prediction_report.pdf`. The report should emphasize how feature selection impacts the model's performance, and the regression results visualization should clearly highlight the relationship between the selected features and the predicted drug response. Ensure the system is designed to be easily extendable for incorporating additional datasets or new features.

## Сравнение: человек / один промпт / агент / браузерный агент

Один промпт и агент по коду — сохранённые базовые оценки из прежних экспериментов.
Браузерный агент — итог данного двухэтапного запуска, включая переход к коду.

| ID | Требование | Человек | Один промпт | Агент по коду | Браузерный агент | Источник браузерного |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Загрузка данных GDSC | FAIL | PASS | PASS | FAIL | выполнение |
| 1 | Отбор признаков | PASS | PASS | PASS | PASS | код |
| 2 | SVM-регрессор | PASS | PASS | PASS | PASS | код |
| 3 | Кросс-валидация | PASS | PASS | PASS | PASS | код |
| 4 | Метрики в results/metrics/ | PASS | FAIL | PASS | FAIL | код |
| 5 | Визуализация в results/figures/ | FAIL | FAIL | PASS | FAIL | код |
| 6 | PDF-отчёт с обязательными разделами | PASS | PASS | PASS | PASS | код |

С человеком совпали 6 из 7 вердиктов.
Функциональный этап: 1 FAIL, 6 UNRESOLVED.
Итог: 3 FAIL, 4 PASS.

## Разбор результата

Программа остановилась с FileNotFoundError: обращение к path_to_gdsc_dataset.csv не нашло опубликованный gdsc_dataset.csv. Функциональный FAIL по загрузке сохранён. Остальные требования получили UNRESOLVED и были оценены по исходникам.

Расхождение с человеком — ID 4: модель поставила FAIL из-за отсутствия файла по точному пути results/metrics/performance.txt, человеческая метка — PASS. Код предусматривает запись по нужному пути, но опубликованный файл лежит в results/performance.txt. Причину человеческого решения по бинарной метке установить нельзя.

PASS по отбору признаков, модели и кросс-валидации подтверждают решение по коду, а не успешное выполнение после загрузки. PASS по PDF ограничен: модель проверяла наличие PDF и содержание соседнего Markdown, а не сам PDF. Полные трассы не переданы.

## Версия и источники

- Коммит браузерного судьи: `87776c1ed35aeed10ce2f1cc754373fe0ef7a00b`.
- Функциональный MLflow run: `182f433322c44ff9b99a9b689e1de847`.
- MLflow run кода: `f79faabde23d457b9235d506efc86959`.
- Базовый судья одним промптом: `062509bc4fcb43be98430081deb9bfe1`.
- Базовый агент по коду: `641fd4f0195c45069517dd5a1ab49067`.

[Исходное сравнение базовых оценок](../../../comparisons/task39-holistic-agent-human/comparison.csv).
Человеческие метки сверены с ранее сохранённой разметкой DevAI OpenHands.

Переданы выдержки трасс, итоговые JSON и метаданные, но не полный конфиг, токены и экспорт MLflow. Внешние run ID не импортированы в нашу локальную базу.
Это отдельные запуски разных версий и сред, а не контролируемая оценка превосходства одного метода.

## Файлы этой задачи

- [comparison.csv](comparison.csv) — полные критерии, четыре оценки, причины и run ID.
- [run.json](run.json) — происхождение, версия и SHA-256 сохранённых артефактов.
- [Функциональные вердикты](artifacts/functional.json).
- [Выдержки свидетельств](artifacts/execution-evidence.json).
- [Итоговые вердикты](artifacts/verdicts.json).
