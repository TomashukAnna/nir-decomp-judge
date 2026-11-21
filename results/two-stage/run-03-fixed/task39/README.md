# Задача 39 — локальный запуск после исправлений

[Все три запуска](../../README.md) · [Вторая задача этого запуска](../task26/README.md)

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

Функциональный FAIL загрузки данных сохранён: Python остановился на ошибочном пути path_to_gdsc_dataset.csv. Остальные требования переданы в код. Инструмент независимых вызовов функций probe был доступен, но модель его не использовала: ограничилась чтением исходников и итоговым списком.

Первый ответ OpenCode содержал синтаксическую ошибку JSON. Валидатор вернул диагностику в ту же сессию, и второй ответ прошёл строгую схему, проверку ID, файлов и строк. Потребовалось одно исправление из трёх возможных — цикл подтверждён реальным запуском.

Расхождение с человеком осталось по ID 4: файл метрик опубликован не по требуемому пути. PASS по последующим этапам получены по коду и не означают успешное выполнение общего конвейера. Проверка содержания PDF также опиралась на соседний Markdown.

## Версия и источники

- Коммит браузерного судьи: `285383d5e8d1e211dd7e3f1f4bd55c9a18adbb14`.
- Функциональный MLflow run: `d230c732c37445ddb58e22487c73fbea`.
- MLflow run кода: `b8f699dbc2a947248a7ccfd162c95cfd`.
- Базовый судья одним промптом: `062509bc4fcb43be98430081deb9bfe1`.
- Базовый агент по коду: `641fd4f0195c45069517dd5a1ab49067`.

[Исходное сравнение базовых оценок](../../../comparisons/task39-holistic-agent-human/comparison.csv).
Человеческие метки сверены с ранее сохранённой разметкой DevAI OpenHands.

Полные наблюдения и трассы находятся локально в full_artifacts из run.json и в MLflow. Здесь сохранены выбранные результаты и выдержки, а не полный журнал.
Это отдельные запуски разных версий и сред, а не контролируемая оценка превосходства одного метода.

## Файлы этой задачи

- [comparison.csv](comparison.csv) — полные критерии, четыре оценки, причины и run ID.
- [run.json](run.json) — происхождение, версия и SHA-256 сохранённых артефактов.
- [Функциональные вердикты](artifacts/functional.json).
- [Выдержки свидетельств](artifacts/execution-evidence.json).
- [Итоговые вердикты](artifacts/verdicts.json).
- [Сырой ответ агента по коду](artifacts/code-raw-response.txt).
- [Ошибка первого JSON](artifacts/code-attempt-1-validation.txt), [сообщение на исправление](artifacts/code-repair-1-prompt.txt), [исправленный ответ](artifacts/code-attempt-2-response.txt).
