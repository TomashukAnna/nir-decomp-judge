# Повтор генерации у руководителя

Ветка `feat/webgen-generation` содержит генератор, конфигурацию OpenCode, Dockerfile
и [полную инструкцию для Linux и Windows](webgen-reproduce.md).

Для воспроизведения текущей проблемы выбрать второе ТЗ test, ID `000002`:

```bash
python3 scripts/prepare_webgen_input.py --ids 000002 --output data/webgen-handoff/input
```

Далее выполнить раздел своей ОС в полной инструкции с Count / `--limit` = 1.
Модель — `iairlab/qwen3.8-27b`, эндпоинт — `https://api.duckduck.cloud/v1`,
ключ берётся из `LITELLM_API_KEY`. Генерацию выполняет OpenCode, а не Python LiteLLM.

Нужен архив исходников созданного приложения и файлов попытки: `run.json`,
`raw-output.jsonl`, `stderr.txt`, `tokens.json`, конфигурация и manifest.
Полный состав результата и исключаемые временные файлы перечислены в инструкции.
Архив с данными в Git не коммитится.

Локальный сбой — `ECONNRESET` при длительной агентной генерации. Последний MLflow
run: `075cac56685e46e4a81d6381e6e754ab`. Просьба выполнить тот же процесс у себя
и вернуть результат, даже если попытка завершится ошибкой. Важен итог в `run.json`:
сам контейнер при сохранённой ошибке может завершиться с exit code 0.
