# Воспроизведение генерации приложений WebGen-Bench

Ветка: `feat/webgen-generation`. Основной сценарий ниже обрабатывает **все 101 ТЗ
из test.jsonl**, создавая отдельное приложение для каждого ID. Нужны Git,
Docker с Linux-контейнерами и Python 3 для
подготовки входа (Python >= 3.10). OpenCode, Python-зависимости и Node входят в Docker-образ;
GPU и локальные веса Qwen не нужны. Исходные данные скачиваются из публичного источника.
Нужен действующий доступ к API выбранной модели: ключ проекта не публикуется.
По умолчанию используется лабораторный эндпоинт; другой OpenAI-compatible
провайдер с поддержкой streaming и tool calling задаётся полями `api_base` и `model`
в конфигурации генерации. Это уже другой эксперимент. Изменённую
конфигурацию нужно сохранить коммитом перед сборкой и использовать новую папку результата.

## Как ТЗ передаётся модели

1. Из WebGen-Bench выбираются только `id` и исходный `instruction`. Приёмочные
   кейсы `ui_instruct`, декомпозиция требований и человеческие оценки не передаются.
2. [webgen_generate.py](../src/decomp_judge/webgen_generate.py) создаёт пустой
   каталог `task-<id>/app/` для каждого ТЗ. Инструкция агента берётся из
   [webgen-builder.md](../configs/webgen-builder.md), пользовательское сообщение:

   ```text
   Implement this specification in the current directory:

   <исходный instruction без изменения текста>
   ```

3. Запускается `opencode run --format json --agent webgen-builder --dir <app> <prompt>`.
   OpenCode вызывает `iairlab/qwen3.8-27b` через OpenAI-compatible API
   `https://api.duckduck.cloud/v1`. В этой генерации Python LiteLLM не вызывается:
   HTTP-запросы делает OpenCode через `@ai-sdk/openai-compatible`.
   `lab/` — локальное имя провайдера OpenCode, а не часть имени модели в API.
4. Это агентный цикл: модель вызывает инструменты чтения, записи и shell,
   OpenCode выполняет их и возвращает результат модели. Агент создаёт исходники,
   устанавливает зависимости при необходимости и проверяет приложение.
   На выходе нужны реальные файлы приложения, не только текстовый ответ модели.

OpenCode закреплён на версии `1.19.4`. Настройки модели и агента:
[webgen-generation.yaml](../configs/webgen-generation.yaml).
`temperature: 0` задана в конфигурации агента. Ограничения числа шагов и общего
времени отключены. [Локальный плагин](../configs/webgen-no-output-cap.mjs)
убирает клиентский лимит ответа OpenCode в 32 000 токенов; пределы сервера и
сетевые тайм-ауты остаются. Изоляция контейнера сохранена.

## Получить код и подготовить ТЗ

```bash
git clone --branch feat/webgen-generation --single-branch https://github.com/TomashukAnna/nir-decomp-judge.git
cd nir-decomp-judge
python3 scripts/prepare_webgen_input.py --all --output data/webgen-all/input
```

В Windows вместо `python3` можно использовать `python`.
Скрипт скачивает `test.jsonl` из коммита
`c89e0743438e458d617cebaf7a682e0e66e41049`, проверяет SHA-256 и сохраняет все ТЗ
в `data/webgen-all/input/test.jsonl`. В `selection.json` остаются источник,
хэши и соответствие ID исходному индексу строки. Уже существующий вход не затирается.
Источник закреплён в [webgen-source.json](../configs/webgen-source.json).

Для совместимости Windows и Linux хэш проверяется после нормализации CRLF → LF.
Если файл уже скачан, вместо обращения в сеть можно добавить
`--source /path/to/test.jsonl`. Для нескольких ТЗ: `--ids 000002 000003 000004`.

Для отдельного пробного запуска можно подготовить другое подмножество:

```bash
python3 scripts/prepare_webgen_input.py --ids 000002 --output data/webgen-single/input
```

Только для этого пробного варианта нужно заменить путь входа на
`data/webgen-single/input/test.jsonl`, выбрать новую папку результата и количество 1.
Основные команды ниже уже настроены на всю выборку.
Количество новых попыток задаётся `--limit N` в Linux или `-Count N` в PowerShell.
Приложения генерируются последовательно. Повторный запуск той же партии с `--resume`
продолжает необработанные ТЗ; успешные исходники не перезаписываются. FAILED и
INTERRUPTED не повторяются автоматически. Для явного повтора нужен отдельный вход
с нужными ID и новая папка результата. Изменять состав уже начатой партии нельзя.

## Linux: собрать и выполнить все ТЗ

Ключ заранее задаётся в окружении как `LITELLM_API_KEY`. Например, в Bash:

```bash
read -r -s -p 'API key: ' LITELLM_API_KEY
export LITELLM_API_KEY
printf '\n'
```

Ввод ключа скрыт. Команды ниже выполняются из корня чистого клона; папка результата
должна быть новой.

```bash
docker build -f docker/webgen.Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t nir-webgen-generator:1.19.4 .

mkdir -p data/generated-apps/webgen-all-001

docker run --name nir-webgen-qwen --init \
  --label org.nir-decomp-judge.purpose=webgen-generation \
  --cpus 2 --memory 4g --pids-limit 256 \
  --cap-drop ALL --security-opt no-new-privileges --read-only \
  --tmpfs /tmp:rw,size=2g \
  --tmpfs /home/generator:rw,uid=1000,gid=1000,size=512m \
  --mount "type=bind,source=$(pwd)/data/webgen-all/input/test.jsonl,target=/input/test.jsonl,readonly" \
  --mount "type=bind,source=$(pwd)/data/generated-apps/webgen-all-001,target=/output" \
  --env LITELLM_API_KEY \
  --env npm_config_cache=/output/.cache/npm \
  --env UV_CACHE_DIR=/output/.cache/uv --env PIP_CACHE_DIR=/output/.cache/pip \
  nir-webgen-generator:1.19.4 \
  --input /input/test.jsonl --output /output --execute --limit 101
```

Для проверки контейнера без обращения к модели в последней строке убрать
`--execute`. Для фонового выполнения добавить `--detach` к `docker run`, журнал:
`docker logs -f nir-webgen-qwen`. Используется одно имя контейнера. Если оно уже
занято завершённым запуском, сохранить его журнал и удалить только остановленный
контейнер перед повтором; папка результата не удаляется.

Для продолжения добавить `--resume` к последней строке, сохранив те же вход,
папку и образ. При трёх ошибках подряд партия остановится раньше окончания
выборки: это защитная остановка, а не ограничение одной задачей. Проверить
`summary.csv` / `status.json` и причину сбоев перед продолжением.

На Linux выходная папка должна быть доступна на запись UID 1000, под которым
работает образ. Если UID пользователя хоста другой, согласовать права на эту
папку; запускать весь контейнер привилегированным не требуется.

## Windows / PowerShell

После той же подготовки входа задать ключ в окружении и выполнить запуск:

```powershell
$secureKey = Read-Host 'API key' -AsSecureString
$env:LITELLM_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
.\scripts\webgen.ps1 -Action Build
.\scripts\webgen.ps1 -Action Start -Count 101 -InputPath data/webgen-all/input/test.jsonl -OutputPath data/generated-apps/webgen-all-001
.\scripts\webgen.ps1 -Action Status -OutputPath data/generated-apps/webgen-all-001
```

Запуск фоновый. Скрипт сам сохраняет журнал завершённого контейнера и пересоздаёт
его под тем же именем `nir-webgen-qwen`. Работающий контейнер не заменяется.
Count / `--limit` — максимум новых попыток, не обещанное число успешных приложений.
Для продолжения использовать `-Action Resume -Count 101` с теми же InputPath и OutputPath.

## Сохранение и передача результатов

Результат находится в `data/generated-apps/webgen-all-001/`.
Приложения лежат в `task-000001/app/` ... `task-000101/app/`.
Сохранить исходники каждого приложения, `manifest.json`, `generation-config.yaml`,
`summary.csv`, `status.json` и из каждой папки `task-<id>/`: `run.json`, `raw-output.jsonl`,
`stderr.txt`, `tokens.json`, `files-sha256.json` (если создан), `prompt.txt`,
`spec.txt`, `config.json`. Для сохранения MLflow — также `mlflow.db` и `mlartifacts/`.
Полезно приложить `data/webgen-all/input/selection.json`.

Передавать архивом, без `.cache`, `runtime`, `node_modules`, `.venv`, `.git` и файлов
с ключами. Эти данные и архив не нужно коммитить в Git. Пути MLflow-артефактов
в базе относятся к контейнеру `/output`; сами артефакты нужно сохранить рядом.

Итог смотреть по `run.json`, не только по exit code контейнера: генератор сохраняет
неудачную попытку как `FAILED` и может штатно завершиться с кодом 0.
`GENERATED` означает сохранённые исходники, а не доказанное выполнение всех требований.

## Границы воспроизводимости

Повторяются закреплённая версия ТЗ, инструкция агента, конфигурация OpenCode,
способ сборки контейнера и сохранения результатов. Версии базовых образов,
OpenCode и Python-зависимостей генератора закреплены в Dockerfile и uv.lock.

Побайтовое совпадение образа и приложений не гарантируется. Пакеты Debian при
сборке и зависимости создаваемых приложений могут обновляться. Модель вызывается
через удалённый сервис, seed модели не передаётся, версия серверного окружения
не закреплена. Значение SEED сохраняется как метаданные запуска. Для независимого
повтора сохранять коммит, конфигурацию, хэши входа, исходники и трассу.

При `ECONNRESET` или других ошибках API попытка может завершиться как FAILED.
Это не означает, что все исходники отсутствуют: промежуточные файлы сохраняются.
Проверять журнал и итог задачи перед продолжением; наличие кода не подтверждает
его функциональную корректность.
