# Запуск двухэтапного судьи на задачах 26 и 39

Ветка `feat/browser-judge`. Архив `browser-judge-inputs.zip` передаётся отдельно:
это входы эксперимента, они не хранятся в Git. Судья сначала проверяет выполнение,
затем передаёт неоценённые требования OpenCode для проверки исходников.

## Подготовка (Linux, bash)

Нужны Python >= 3.12, uv, Node.js с npm и OpenCode **1.19.4**.
Все команды ниже выполняются из корня репозитория. При новом клонировании:

```bash
git clone --branch feat/browser-judge --single-branch https://github.com/TomashukAnna/nir-decomp-judge.git
cd nir-decomp-judge
uv sync --locked
```

Распакуйте архив в `data/handoff/`. Должны появиться
`data/handoff/task26/{app,spec.txt,requirements.json}` и такая же структура для task39.
`manifest.json` в корне архива содержит SHA-256 каждого файла.
Копии app предназначены только для чтения; зависимости устанавливаются отдельно:

```bash
mkdir -p data/handoff-runtime
cp -R data/handoff/task26/app data/handoff-runtime/task26
cp -R data/handoff/task39/app data/handoff-runtime/task39
npm --prefix data/handoff-runtime/task26 install --no-audit --no-fund
uv run --with playwright==1.52.0 python -m playwright install --with-deps chromium
export OPENCODE_BIN="$(command -v opencode)"
"$OPENCODE_BIN" --version
```

Команды рассчитаны на новые каталоги runtime и результатов. На кластере браузеру
нужны системные библиотеки и разрешённый запуск локального HTTP-сервера.
Установка `--with-deps` может требовать администратора; при готовом окружении
достаточно `python -m playwright install chromium` с тем же `uv run --with`.

## API

Ключ должен находиться в `LITELLM_API_KEY`. Его не включаем в архив и результаты.
По умолчанию используется лабораторный API из `opencode.json`.
Для внутреннего API создайте игнорируемую копию конфигурации:

```bash
cp opencode.json data/handoff-opencode.json
```

В ней замените `provider.lab.options.baseURL`, ключ модели в `provider.lab.models`,
а также `model` и `small_model` (`lab/<имя модели на сервере>`).
`apiKey` оставьте `{env:LITELLM_API_KEY}`.
В поле `agent.requirements-judge.prompt` укажите абсолютный путь к
`configs/judge-agent.md` в этом checkout, поскольку копия конфигурации лежит в data.
Один файл конфигурации используется функциональным этапом через Python LiteLLM
и этапом чтения кода через OpenCode. Нужен OpenAI-совместимый API.

Рабочее дерево кода судьи должно быть чистым; копия конфига и входы в data игнорируются.
Не редактируйте исходные приложения и не исправляйте ошибки задачи 39 перед запуском.

## Запуск

Если внутренний API не нужен, замените `--config data/handoff-opencode.json`
на `--config opencode.json`. Для каждого нового запуска используйте новый output.

```bash
uv run --with playwright==1.52.0 --with pandas --with scikit-learn --with matplotlib --with seaborn \
  python -m decomp_judge.browser_judge \
  --config data/handoff-opencode.json \
  --app data/handoff/task26/app --runtime-app data/handoff-runtime/task26 \
  --spec data/handoff/task26/spec.txt --requirements data/handoff/task26/requirements.json \
  --opencode "$OPENCODE_BIN" --output data/handoff-runs/task26-internal-001

uv run --with playwright==1.52.0 --with pandas --with scikit-learn --with matplotlib --with seaborn \
  python -m decomp_judge.browser_judge \
  --config data/handoff-opencode.json \
  --app data/handoff/task39/app --runtime-app data/handoff-runtime/task39 \
  --spec data/handoff/task39/spec.txt --requirements data/handoff/task39/requirements.json \
  --opencode "$OPENCODE_BIN" --output data/handoff-runs/task39-internal-001
```

Дополнительные Python-пакеты нужны приложению задачи 39; это зависимости среды
прогона, не изменение исходников. Их точные версии разрешаются uv при подготовке окружения.

В задаче 39 опубликованный код обращается к `path_to_gdsc_dataset.csv`.
Ошибка загрузки ожидаема; агент должен оценить наблюдение и, где возможно,
проверить независимые компоненты, затем перейти к исходникам для оставшихся требований.

До завершения обоих запусков результаты остаются в игнорируемой папке `data/`:
это необходимо, поскольку каждый этап требует чистого Git checkout.
Затем скопируйте их для передачи:

```bash
mkdir -p results/two-stage
cp -R data/handoff-runs/task26-internal-001 results/two-stage/
cp -R data/handoff-runs/task39-internal-001 results/two-stage/
```

## Что вернуть

Полные каталоги `results/two-stage/task26-internal-001` и `task39-internal-001`,
включая `functional.json`, `observations.json`, итоговый `verdicts.json` и подкаталог
`code/` с ответами и трассой OpenCode. При сбое — всё, что успело сохраниться,
и traceback команды. Приложите `git rev-parse HEAD`, версию OpenCode,
имя модели и версии среды; ключ не прикладывайте.

MLflow пишет в локальное хранилище согласно настройкам проекта. Run ID функционального
этапа и этапа кода записываются в итоговом JSON, если оба этапа завершились.
Сравнение с разметкой и анализ результатов выполним после получения файлов.

Приложения запускаются в изолированном окружении: отдельная runtime-копия не заменяет
контейнер или ограничения вычислительного узла. Ошибка API сама по себе не означает FAIL
по требованиям приложения.

