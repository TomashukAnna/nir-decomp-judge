# Быстрый запуск генерации всей выборки

Ветка `feat/webgen-generation` содержит генератор, конфигурацию OpenCode, Dockerfile
и [полную инструкцию для Linux и Windows](webgen-reproduce.md).

Генератор создаёт по приложению для **всех 101 ТЗ** из test.jsonl WebGen-Bench.
Подготовка всей выборки:

```bash
git clone --branch feat/webgen-generation --single-branch https://github.com/TomashukAnna/nir-decomp-judge.git
cd nir-decomp-judge
python3 scripts/prepare_webgen_input.py --all --output data/webgen-all/input
```

Нужны Git, Docker с Linux-контейнерами, Python >= 3.10 и доступ к API модели.
В Windows можно использовать `python` вместо `python3`.

Далее выполнить раздел своей ОС в полной инструкции с Count / `--limit` = 101.
Генерация последовательная, у каждого ТЗ собственная папка `task-<id>/app/`.
Модель — `iairlab/qwen3.8-27b`, эндпоинт — `https://api.duckduck.cloud/v1`,
ключ берётся из `LITELLM_API_KEY`. Генерацию выполняет OpenCode, а не Python LiteLLM.

В папке результата сохраняются исходники приложений и файлы каждой попытки: `run.json`,
`raw-output.jsonl`, `stderr.txt`, `tokens.json`, конфигурация и manifest.
Полный состав результата перечислен в инструкции. При передаче результатов
архивировать исходники и метаданные без временных файлов и секретов.
Данные и архивы с приложениями в Git не коммитятся.

Итог каждой попытки смотреть в `run.json`: сам контейнер при сохранённой ошибке
может завершиться с exit code 0. Работа требует действующего доступа к API модели.
Повторяется процесс генерации; точное совпадение сгенерированных файлов не гарантируется.
