# Сводка человеческой разметки DevAI

Источник: https://github.com/metauto-ai/agent-as-a-judge
Коммит: `43c2fa7b84ef0c1522e8250576e6798bace29dd0`.

Подсчёт опубликованных human_as_a_judge без исправления исходных данных. Preferences исключены.

| Фреймворк | Задач | Меток требований | true | false | Задач с опубликованным satisfied_all_requirements=true | Задач, где все отдельные метки true |
| --- | --- | --- | --- | --- | --- | --- |
| OpenHands | 55 | 366 | 157 | 209 | 2 | 1 |
| MetaGPT | 55 | 366 | 81 | 285 | 0 | 0 |
| GPT-Pilot | 55 | 366 | 163 | 203 | 5 | 5 |

Исходные instances: 55 задач, 365 требований. Нельзя автоматически считать, что разметка каждого фреймворка содержит ровно те же 365 критериев (см. расхождение 1).

## Обнаруженные расхождения (сортировка по критичность)

1. **OpenHands/MetaGPT/GPT-Pilot, 52_Devin_AI_Trains_an_AI**: В описании задачи ID  [0, 1, 2, 3], ID в разметке [0, 1, 2, 3, 4]. Метки не переназначались.
2.  **OpenHands, 31_Cancer_Prediction_SVM_BreastCancer_ML**: satisfied_all_requirements=True, при этом ID: [5] = false.
3. **OpenHands**: 23 текстов criteria различаются при одинаковом ID (без нормализации текста).
4. **MetaGPT**: 24 текстов criteria различаются при одинаковом ID (без нормализации текста).
5. **GPT-Pilot**: 24 текстов criteria различаются при одинаковом ID (без нормализации текста).
6. **OpenHands, 13_Style_Transfer_Perceptual_Loss_CustomImages_DL**: Название в описании задачи (*instances*) "13_Style_Transfer_Perceptual_Loss_CustomImages_DL", а в разметке (*human_as_a_judge*) "13_Style_Transfer_Perceptual_Loss_CustomImages_DL.json".
7.  **GPT-Pilot, 13_Style_Transfer_Perceptual_Loss_CustomImages_DL**: поле name в разметке = "13_Style_Transfer_Perceptual_Loss_CustomImages_D" (потерялась L на конце)

 ## Вывод:
 Два существенных несоответствия:
- В исходных задачах 365 требований, а в разметке каждого фреймворка — 366. У задачи №52 в описании четыре требования, а в разметке — пять; отличаются и формулировки. Простого соединения по ID здесь недостаточно.
- У OpenHands в задаче №31 итоговый флаг равен true, хотя требование №5 отмечено false. Если считать полное выполнение по отдельным меткам, получится 1 задача вместо 2.

Также встретились различия текстов критериев и опечатки в названиях задач.

