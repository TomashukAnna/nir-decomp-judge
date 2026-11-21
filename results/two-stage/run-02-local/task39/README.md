# Задача 39 — первый локальный запуск, до исправлений

[Все три запуска](../../README.md) · [Вторая задача этого запуска](../task26/README.md)

Дата: 19 ноября 2026. Сравнение ниже относится только к этому запуску браузерного судьи.

## Описание задачи

Система прогнозирования ответа на лекарственные препараты по данным GDSC с помощью SVM-регрессора. Нужно загрузить данные, отобрать признаки, обучить модель и оценить её кросс-валидацией. Метрики сохраняются в results/metrics/performance.txt, визуализация seaborn — в results/figures/, PDF-отчёт — в results/drug_response_prediction_report.pdf. Отчёт должен объяснять влияние отбора признаков и показывать связь признаков с прогнозом. Система должна допускать добавление наборов данных и признаков. Источник: задача 39 DevAI, опубликованное приложение OpenHands.

### Исходное ТЗ

> Develop a system to predict drug response using the GDSC dataset with a Support Vector Machine (SVM) regressor. Load the dataset and perform feature selection to identify key features in `src/data_loader.py`. Implement the SVM regressor in `src/model.py`. Use cross-validation to evaluate the model's performance in `src/train.py`.  Save the performance results to `results/metrics/performance.txt`. Visualize the regression results using seaborn and save it under `results/figures/`. Next, create a report including the data preprocessing, model training, evaluation process, and the visualization. Save the report as `results/drug_response_prediction_report.pdf`. The report should emphasize how feature selection impacts the model's performance, and the regression results visualization should clearly highlight the relationship between the selected features and the predicted drug response. Ensure the system is designed to be easily extendable for incorporating additional datasets or new features.

## Сравнение: человек / один промпт / агент / браузерный агент

Один промпт и агент по коду — сохранённые базовые оценки из прежних экспериментов.
Браузерный агент — итог данного двухэтапного запуска, включая переход к коду.

| ID | Требование | Человек | Один промпт | Агент по коду | Браузерный агент | Источник браузерного |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Загрузка данных GDSC | FAIL | PASS | PASS | Нет ответа | — |
| 1 | Отбор признаков | PASS | PASS | PASS | Нет ответа | — |
| 2 | SVM-регрессор | PASS | PASS | PASS | Нет ответа | — |
| 3 | Кросс-валидация | PASS | PASS | PASS | Нет ответа | — |
| 4 | Метрики в results/metrics/ | PASS | FAIL | PASS | Нет ответа | — |
| 5 | Визуализация в results/figures/ | FAIL | FAIL | PASS | Нет ответа | — |
| 6 | PDF-отчёт с обязательными разделами | PASS | PASS | PASS | Нет ответа | — |

Совпадение с человеком не вычисляется для невалидного ответа.
Функциональный этап: 7 UNRESOLVED.
Итоговых вердиктов нет: ответ агента отклонён.

## Разбор результата

Ошибка загрузки данных наблюдалась при запуске Python, но не стала принятым функциональным FAIL. Модель несколько раз вернула обычный текст вместо JSON, а последний ответ оказался пустым после удаления пробелов. Все требования ушли в код.

Ответ агента по коду отклонён целиком из-за лишнего поля `line_start_note: null` в свидетельстве требования 6. Использованный коммит не содержал цикл трёх исправлений из более позднего main. Поэтому диагностика не была возвращена агенту.

Валидного итога нет. «Нет ответа» в таблице означает невалидный ответ судьи, а не FAIL или NA требований приложения. Совпадение с человеком для этого запуска не вычисляется. Непринятый сырой ответ сохранён как диагностика, а не как итоговая разметка.

## Версия и источники

- Коммит браузерного судьи: `87776c1ed35aeed10ce2f1cc754373fe0ef7a00b`.
- Функциональный MLflow run: `92739c0276f7414e9efe01df6a9f4578`.
- MLflow run кода: `3000a1308551450f8119fd4bb11d8a78`.
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
- [Ошибка агента по коду](artifacts/code-error.json).
- [Сырой ответ агента по коду](artifacts/code-raw-response.txt).
