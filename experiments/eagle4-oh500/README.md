# EAGLE-4 OH500: 500 полных траекторий OpenHands · Qwen3.8-27B

Подвыборка [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) v1.2 (ревизия `4fa4c2ae9afc6b397bc50391d43450c79993c138`) для первого прогона EAGLE-4 на Qwen3.8-27B: 500 траекторий OpenHands, по 100 в каждой из 5 корзин длины контекста. Траектории полные, без нарезки; к ним приложена разметка токенов для выбора точек замера.
Модели и GPU не запускались. Основа — аудит [`../open-swe-traces-audit/README.md`](../open-swe-traces-audit/README.md) (§2–§5.1).

## TL;DR

- **500 траекторий OpenHands** (282 scale-swe, 218 swe-rebench-v2): ровно по 100 в корзинах 32K–64K, 64K–100K, 100K–128K, 128K–200K и 200K–256K. Пул отбора — 11 048 траекторий.
- Внутри: **41 753 вызова модели** и 45 949 ответов инструментов.
- `L_max` от 33.2K до 258.1K (median 113K); вызовов на траекторию 18–225 (median 73).
- 348 репозиториев: не больше 11 на один, у 267 по одной траектории. Python — 70%, всего 9 языков.
- **Реконструкция согласована по всем вызовам, не по выборке.**
  - Для каждого из 41 753 вызовов токены `render(messages[:j], add_generation_prompt=True)` совпали со срезом полного рендера.
  - `input_len` совпал с `input_visible_template` аудита у всех 41 753 вызовов.
  - Пустого reasoning в выборке нет, так что известного артефакта BPE тоже нет.
- **Оригинальные строки** по значениям и схеме идентичны строкам shards (500/500).
- Повторный отбор с тем же seed даёт ту же выборку. Полный повторный прогон `select` + `export` даёт файлы с теми же sha256 (§5).
- **Вход вызова `q`** = `ids[:call_start] + GEN_IDS`: `ids` из `render_ids.parquet`, `call_start` из `calls.parquet`. Там же разметка конца reasoning, tool calls, конца хода и ответов инструментов (§4).
- **Token ids генерации в датасете нет**: ни token ids, ни logprobs, ни usage. `ids` здесь — наш рендер. Начала вызовов от этого не страдают. В точках внутри хода текст до точки точный, а токены — каноническое разбиение, не обязательно то, что сэмплировала модель (§6).
- Подготовка шла на beleriand (72 процесса): export 6 мин, verify 40 с. Модели не запускались.

## 1. Отбор

| Фильтр | Осталось траекторий |
|---|---|
| весь v1.2 | 235 391 |
| `config == "openhands"` (оба источника) | 25 599 |
| `resolved == 1` | 11 933 |
| `finish_kind == "finish"` (агент завершил штатно) | 11 933 — у OpenHands так заканчиваются все |
| `L_max_best_estimate` ∈ [32 768, 262 144) | **11 048** |

`L_max` — максимум по вызовам траектории длины входа **одного** вызова: рендер `messages[:j]` через шаблон + generation prompt. У OpenHands `L_max_best_estimate == L_max_visible_template`: сжатия нет, прошлый reasoning остаётся в контексте.

| `bin` | Корзина | Границы `L_max` | Пул | Выборка | Доля пула | scale-swe / rebench-v2 | `L_max` median | Вызовов на траекторию, median | Вызовов всего |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 32K–64K | [32 768, 65 536) | 5 087 | 100 | 2.0% | 64 / 36 | 51.9K | 38 | 3 984 |
| 1 | 64K–100K | [65 536, 100 000) | 3 467 | 100 | 2.9% | 55 / 45 | 79.9K | 56 | 5 617 |
| 2 | 100K–128K | [100 000, 131 072) | 1 428 | 100 | 7.0% | 63 / 37 | 113.2K | 75 | 7 667 |
| 3 | 128K–200K | [131 072, 200 000) | 917 | 100 | 10.9% | 50 / 50 | 148.7K | 100 | 10 208 |
| 4 | 200K–256K | [200 000, 262 144) | 149 | 100 | 67.1% | 50 / 50 | 224.7K | 140 | 14 277 |
| | **всего** | | **11 048** | **500** | | 282 / 218 | 113.2K | 73 | **41 753** |

Подписи следуют аудиту: 64K = 65 536, 128K = 131 072, 256K = 262 144, а 100K и 200K — десятичные. Остальные распределения (языки, категории, репозитории, квантили длин входа и выхода) — в [`distributions.md`](distributions.md).

- **Внутри корзины** траектории ранжируются по `sha256(f"{seed}:{trajectory_id}")`, и берутся ранги 0–99, `seed = 20261116`. Это равномерная выборка без возвращения. Она не зависит от порядка строк во входной таблице и от версии numpy.
- Первые k рангов корзины — тоже равномерная подвыборка. Для пилота можно взять `rank_in_bin < 20`.
- Пулы совпали с цифрами аудита (5 087 / 3 467 / 1 428 / 917 / 149).
- Повторный отбор с тем же seed воспроизводит выборку побайтно (§5).
- **Лимита на репозиторий нет**: распределение репозиториев естественное. `repo` сохранён, чтобы при анализе считать доверительные интервалы с кластеризацией по репозиторию.
- «Одна траектория на задачу» выполняется автоматически: в OpenHands на задачу один прогон.
- **Веса.** Доля выборки от пула в корзинах разная: от 2% до 67%. Оценки внутри корзины не требуют весов. Для оценки «по всем траекториям ≥32K» корзину нужно взвешивать на `pool / 100`.

## 2. Почему так

- **OpenHands — самая чистая конфигурация аудита.**
  - С учётом прошлого reasoning распределение `L_max` обрывается ровно у окна 262 144 токенов, и ни одна траектория его не превышает (на обоих источниках). Значит, прошлый reasoning был в контексте, и рендер шаблоном с сохранённым reasoning совпадает с тем, что видела модель.
  - Один фиксированный system prompt: префикс system+tools — 3 938 токенов. Четыре инструмента: `execute_bash`, `str_replace_editor`, `think`, `finish`.
  - Нет сообщений об ошибках формата, незавершённых прогонов и сжатия контекста.
  - Самые длинные контексты среди чистых конфигураций (median `L_max` 78K, p95 180K).
- **`resolved = 1`** — выбор пользователя для первого прогона.
- **Стратификация по длине нужна**, потому что равномерная выборка дала бы около 7 траекторий в 200–256K.
- **Траектории полные, без нарезки.** Точки замера (начало вызова, после ответа инструмента, после reasoning и т. д.) выбираются позже по разметке (§4).

## 3. Файлы

Данные лежат в `data/open-swe-traces/eagle4_oh500/` (каталог исключён из Git). Копия на beleriand: `~/specdec-open-swe/data/open-swe-traces/eagle4_oh500/`.

| Файл | Строк | Размер | Что внутри |
|---|---|---|---|
| `trajectories.parquet` | 500 | 60 MB | оригинальные строки + происхождение, отбор, параметры рендера, метрики аудита |
| `calls.parquet` | 41 753 | 2.0 MB | разметка и проверки по вызовам |
| `tool_responses.parquet` | 45 949 | 1.2 MB | разметка ответов инструментов |
| `render_ids.parquet` | 500 | 77 MB | token ids полного рендера каждой траектории |

sha256 файлов записаны в `manifest.json` → `outputs`.

**`trajectories.parquet`**: 500 строк, отсортированы по `(bin, rank_in_bin)`, row group по 20 строк.

| Группа | Колонки |
|---|---|
| оригинал, без изменений | `instance_id`, `repo`, `license`, `language`, `trajectory_id`, `messages`, `tools`, `resolved`, `metadata` (`category`, `teacher_model`, `reference_patch`, `model_patch`), `hf_dataset_name`; схема та же, что в shards |
| происхождение | `src_dataset`, `src_config`, `src_revision`, `src_harness`, `src_source`, `src_path` (`data/openhands/qwen38_27b/<source>/<file>`), `src_row` (номер строки в файле) |
| отбор | `bin` (0–4), `bin_label`, `bin_lo`, `bin_hi`, `L_max`, `seed`, `rank_in_bin`, `sample_key` |
| параметры рендера | `render_tokenizer`, `render_tokenizer_revision`, `render_chat_template_sha256`, `render_reasoning_effort` (`xhigh`), `render_enable_thinking` (True), `render_preserve_thinking` (True), `render_context_mode` (`preserve`), `render_tools_openai` (схемы в формате OpenAI, JSON-строки) |
| результаты рендера | `render_n_tokens`, `render_n_calls`, `render_L_max`, `render_L_max_call`, `render_session_prompt_tokens_sum`, `render_output_tokens_max`, `render_prefix_system_tools`, `render_n_tool_msgs`, `render_n_binding_mismatch`, `render_last_tool`, `render_text_sha256`, `render_ids_sha256`, `render_calls_prefix_ok`, `render_calls_slice_ok` |
| метрики аудита | `audit_*` — все колонки `audit/final/trajectories.csv` (`audit_n_llm_calls`, `audit_session_prompt_tokens_sum`, `audit_tool_output_tokens_*`, `audit_category`, …) и `termination.csv` (`audit_finish_kind`, `audit_last_tool`, `audit_rep_last`). Колонки, которые дублируют оригинальные (`instance_id`, `repo`, `language`, `resolved`, `category`), проверены на равенство |

**`calls.parquet`**: одна строка на вызов модели.

| Колонка | Смысл |
|---|---|
| `trajectory_id`, `bin`, `call`, `n_calls`, `is_last`, `msg_index` | вызов `call` (с 0) — assistant-сообщение `messages[msg_index]` |
| `call_start` | индекс `<\|im_start\|>` этого assistant-хода в полном рендере |
| `input_len` | длина входа вызова = `call_start + 5` |
| `input_len_audit`, `output_len_audit`, `msg_index_audit_eq` | значения из `calls.csv.gz` аудита для сверки |
| `think_end` | индекс `</think>` |
| `tool_call_starts`, `tool_call_ends` | индексы всех `<tool_call>` и `</tool_call>` хода |
| `turn_end` | индекс `<\|im_end\|>`, закрывающего ход; `output_len = turn_end + 1 − input_len` |
| `n_reasoning_tokens` | `think_end − input_len` (reasoning + перевод строки перед `</think>`) |
| `n_tool_calls`, `tool_names`, `tool_call_ids` | из `messages[msg_index].tool_calls` |
| `reasoning_chars`, `content_chars`, `reasoning_empty` | |
| `prefix_str_eq` | строка `render(messages[:msg_index], add_generation_prompt=True)` — префикс полного рендера |
| `prefix_ids_eq_hist_gen` | токены этого префикса == `ids[:call_start] + GEN_IDS` |
| `prefix_ids_eq_slice` | токены этого префикса == `ids[:input_len]` (проверка из `check_exact.py`) |

**`tool_responses.parquet`**: одна строка на tool-сообщение. Колонки: `trajectory_id`, `bin`, `msg_index`; `call` и `k` — ответ на `k`-й tool call вызова `call`, привязка по позиции; `tool_name` и `tool_call_id` — того tool call; `resp_start` и `resp_end` — индексы `<tool_response>` и `</tool_response>`; `n_tokens = resp_end − resp_start − 1`; `group_start` — `<|im_start|>` user-хода, в котором шаблон склеивает подряд идущие ответы; `content_chars`.

**`render_ids.parquet`**: `trajectory_id`, `bin`, `n_tokens`, `input_ids` (list<int32>) — token ids полного рендера, к которым относятся все индексы. Их можно получить заново рендером (§4), здесь они сохранены для удобства и как эталон.

## 4. Рендер и разметка

Рендер тот же, что в аудите:

- `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`: `chat_template.jinja` и `tokenizer.json`, sha256 в `manifest.json`;
- `apply_chat_template(messages, tools=tools, reasoning_effort="xhigh", enable_thinking=True, preserve_thinking=True)`, затем `tokenizer(text, add_special_tokens=False)`;
- `arguments` у tool calls разбираются из JSON-строки в dict, как это делает vLLM (`audit.conv_msgs`);
- `tools` уже хранятся в формате OpenAI (`audit.conv_tools` ничего не меняет).

Полный рендер траектории (`add_generation_prompt=False`) — это последовательность ходов:

```
<|im_start|>system … # Tools … <tools>…</tools> … system prompt<|im_end|>\n          ← 3 938 токенов у всех
<|im_start|>user\n<задача><|im_end|>\n
<|im_start|>assistant\n<think>\n<reasoning>\n</think>\n\n<content>[\n\n]<tool_call>…</tool_call>[\n<tool_call>…</tool_call>]<|im_end|>\n
<|im_start|>user\n<tool_response>\n<вывод 1>\n</tool_response>\n<tool_response>\n<вывод 2>\n</tool_response><|im_end|>\n
<|im_start|>assistant\n<think>\n …
```

Все маркеры (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`) — отдельные added tokens. Индексы разметки указывают ровно на них.
Позиции вычисляются **по структуре шаблона**: длины рендеров префиксов `messages[:j]` с generation prompt и без него. Поиска по тексту нет, так что литералы вроде `<|im_end|>` внутри выводов разметку не сбивают. Каждая позиция проверяется на ожидаемый token id.

**Точки замера.** `GEN_IDS = [248045, 74455, 198, 248068, 198]` — это `"<|im_start|>assistant\n<think>\n"`.

| Точка | Контекст (`prompt_token_ids`) | Что генерирует модель дальше | Реальный вход? | Токены до точки |
|---|---|---|---|---|
| начало вызова `q` | `ids[:call_start] + GEN_IDS` (длина `input_len`) | reasoning … `<\|im_end\|>` | **да** | точно те, что видела модель |
| после reasoning | `ids[:think_end + 1]` (кончается на `</think>`) | `\n\n` + content / tool calls | нет, продолжение генерации | текущий ход — каноническое разбиение текста |
| перед `k`-м tool call | `ids[:tool_call_starts[k]]` | `<tool_call>…` | нет | то же |
| после `k`-го tool call, не последнего | `ids[:tool_call_ends[k] + 1]` (кончается на `</tool_call>`) | `\n<tool_call>…` (следующий параллельный вызов) | нет | то же |
| после последнего tool call | `ids[:tool_call_ends[-1] + 1]` | только `<\|im_end\|>`, точка бесполезна | нет | |
| после ответа инструмента | это начало следующего вызова (`call_start` вызова `q+1`) | | да | точно |

Подавайте `ids` напрямую как `prompt_token_ids`, без чат-шаблона и без generation prompt. `</think>`, `<tool_call>` и `</tool_call>` — одиночные служебные токены, поэтому границы точек точные. Почему токены внутри хода канонизированы — см. §6.

Пример:

```python
import pyarrow.parquet as pq
D = "data/open-swe-traces/eagle4_oh500"
GEN_IDS = [248045, 74455, 198, 248068, 198]
tid = pq.read_table(f"{D}/trajectories.parquet", columns=["trajectory_id"]).column(0)[0].as_py()
ids = pq.read_table(f"{D}/render_ids.parquet", filters=[("trajectory_id", "=", tid)]).column("input_ids")[0].as_py()
calls = pq.read_table(f"{D}/calls.parquet", filters=[("trajectory_id", "=", tid)]).to_pandas()
c = calls.iloc[10]
prompt_ids = ids[:c.call_start] + GEN_IDS          # точный вход вызова 10, len == c.input_len
reference = ids[c.input_len:c.turn_end + 1]        # канонические токены его ответа (reasoning … <|im_end|>)
```

`ids[:input_len]` равно `ids[:call_start] + GEN_IDS` у всех вызовов, кроме тех, где reasoning пуст. Там в полном рендере `<think>\n\n</think>` даёт один токен `\n\n` (271) вместо `\n` (198). Это тот же артефакт BPE, что в аудите (§4.1). Поэтому вход вызова лучше собирать как `ids[:call_start] + GEN_IDS`.

## 5. Проверки

Источники: `export_checks.json` (считается при экспорте) и `verify.json`. `verify` — отдельный проход по готовым файлам со свежим токенайзером.
Данные и повторный прогон сделаны версией `build.py` с sha256 `9a53d00a…`. Текущая версия (sha256 в `manifest.json`) отличается от неё только текстом оговорок в `report`.

| Проверка | Результат |
|---|---|
| по 100 в корзине, 500 уникальных `trajectory_id` | ✅ |
| все `resolved == 1`, `finish_kind == "finish"`, `config == "openhands"`; `teacher_model`: `xhigh` и thinking | ✅ 500/500 |
| `L_max` в границах своей корзины и равен `L_max` нового рендера | ✅ 500/500 |
| оригинальные колонки равны строкам shards по `src_path` / `src_row`, включая схему | ✅ 500/500 |
| повторный отбор с тем же seed | ✅ те же id, корзины, ранги и порядок |
| повторный полный прогон `select` + `export` в отдельный каталог | ✅ sha256 совпали у `selection.csv` и у всех 4 parquet (`manifest.json` → `outputs`) |
| `input_len == input_visible_template`: `audit/final/calls.csv.gz` (Mac) и `audit/full/calls.csv.gz` (beleriand) | ✅ 41 753 / 41 753 в обоих |
| `output_len == output_template`, `msg_index` совпадает | ✅ 41 753 / 41 753 |
| строка `render(messages[:j], add_generation_prompt=True)` — префикс полного рендера | ✅ 41 753 / 41 753 |
| tokens(этого префикса) == `ids[:call_start] + GEN_IDS` | ✅ 41 753 / 41 753 |
| tokens(этого префикса) == `ids[:input_len]` (как в `check_exact.py`, но для **всех** вызовов) | ✅ 41 753 / 41 753 (пустого reasoning нет) |
| по траекториям совпадают с аудитом: число вызовов, длина рендера, сумма входов по сессии, max выхода, префикс system+tools (3 938), индекс вызова с `L_max`, число tool-сообщений | ✅ 500/500 по каждой |
| привязка по позиции: после хода с N tool calls идут ровно N ответов (кроме финального `finish`) | ✅ 0 нарушений; у каждого ответа есть свой tool call |
| новый рендер с нуля == сохранённые `input_ids`; каждый индекс разметки указывает на ожидаемый token id; ходы упорядочены; последний ход заканчивается в конце рендера | ✅ 500/500 |
| последний вызов каждой траектории — `finish` | ✅ 500/500 |

## 6. Оговорки

- **Sampling неизвестен.** В данных нет параметров sampling, `max_tokens` и per-call usage. `generation_config` модели: T=1.0, top_k=20, top_p=0.95. Были ли прогоны OpenHands greedy, неизвестно; mini точно не greedy (аудит §2).
- **Версия OpenHands** — v0.53.0 со слов автора датасета для v1.0, для v1.2 не подтверждена.
- **Привязка ответа инструмента к вызову — по позиции**: у tool-сообщения нет `tool_call_id`. В этой выборке у каждого хода с N tool calls ровно N ответов подряд, кроме последнего `finish`.
- **Неудачные ответы модели** (без tool call) при конверсии датасета удалялись у mini-SWE-agent. У OpenHands следов этого нет, но исключить это нельзя.
- **Token ids генерации не сохранены.** В Open-SWE-Traces v1.2 нет ни token ids, ни logprobs, ни usage. Проверено:
  - у всех 100 shards одна схема: `messages{role, content, reasoning_content, tool_calls{function{name, arguments}, id, type}}`, `tools`, `resolved`, `metadata`, `hf_dataset_name`;
  - в дереве репозитория (563 пути) таких файлов нет;
  - в карточке и в `data_distribution.md` они не упоминаются.

  `tool_calls[].id` (`chatcmpl-tool-…`) — это id вызова от vLLM, а не token ids. `input_ids` в `render_ids.parquet` — наш рендер текста. Что из этого следует:
  - **Начала вызовов точные.** Harness передаёт историю текстом, а сервер заново рендерит её и токенизирует детерминированно, то есть канонически. Модель видела ровно `ids[:call_start] + GEN_IDS`.
  - **Точки внутри хода канонизированы.** Это после `</think>`, перед или после `<tool_call>`. В реальном роллауте в KV-кэше к этому моменту лежали токены, которые сэмплировала модель, а она могла выдать неканоническое разбиение того же текста (`hel` + `lo` вместо `hello`). Текст до точки точный, служебные токены-границы тоже. Но токены reasoning и аргументов текущего хода — каноническое разбиение. Восстановить исходное по данным нельзя, долю неканонических токенов оценить тоже.
  - **Реальные входы модели — только начала вызовов.** Точки внутри хода — продолжения сгенерированного текста.
- **Длины нельзя сверить с сервером**: usage нет, все длины получены рендером шаблона.
- **Выборка не представляет все длинные траектории.** Среди `L_max ≥ 128K` по корпусу решено лишь 24%. Решённые траектории могут отличаться от нерешённых: короче reasoning, меньше повторов.
- **Корзины взвешены неравномерно** (§1): для оценок по всему пулу нужны веса `pool / 100`.

## 7. Воспроизведение

На beleriand (shards, токенайзер и venv аудита уже там). `audit/final/trajectories.csv` загружен туда с Mac (sha256 `ec63b4e9…`, тот же файл). `termination.csv` идентичен `audit/full/termination.csv`.

```bash
cd ~/specdec-open-swe
D=$PWD/data/open-swe-traces; E=$PWD/eagle4-oh500/repo/experiments/eagle4-oh500   # копия experiments/ из репозитория
PY=(.venv/bin/python -I)
$PY $E/build.py select --trajectories $D/audit/final/trajectories.csv --termination $D/audit/final/termination.csv
$PY $E/build.py export --shards $D/shards --tok $D/tokenizer/qwen3.8-27b@1d4bf0f2ff \
    --audit-calls $D/audit/full/calls.csv.gz --out $D/eagle4_oh500 --workers 72          # ~6 мин на 72 ядрах
$PY $E/build.py verify --trajectories $D/audit/final/trajectories.csv --termination $D/audit/final/termination.csv \
    --shards $D/shards --tok $D/tokenizer/qwen3.8-27b@1d4bf0f2ff --out $D/eagle4_oh500 --workers 72   # ~40 с
$PY $E/build.py report --tok $D/tokenizer/qwen3.8-27b@1d4bf0f2ff --out $D/eagle4_oh500
```

Обратно на Mac:

```bash
rsync -av beleriand:specdec-open-swe/data/open-swe-traces/eagle4_oh500/ data/open-swe-traces/eagle4_oh500/
rsync -av beleriand:specdec-open-swe/eagle4-oh500/repo/experiments/eagle4-oh500/ experiments/eagle4-oh500/
```

Локально всё тоже работает (`.venv-mac`, те же версии пакетов), но нужны все 12 shards OpenHands (3.54 GB).

## 8. Файлы в этом каталоге

- `build.py` — отбор, экспорт, разметка, проверки, отчёт (`select` / `export` / `verify` / `report`). Рендер берёт `conv_msgs` / `conv_tools` из `../open-swe-traces-audit/audit.py`.
- `selection.csv` — 500 строк: корзина, ранг, ключ и все метрики аудита.
- `pool_counts.json` — воронка, пулы по корзинам и источникам.
- `export_checks.json`, `verify.json` — результаты проверок.
- `manifest.json` — ревизии датасета и токенайзера, sha256 токенайзера, параметры рендера, фильтры, корзины, seed, размеры пулов и выборки, обоснование, оговорки, sha256 скриптов, входов и выходов, версии пакетов.
- `distributions.md`, `distributions.json` — распределения выборки: корзины, источники, языки, категории, репозитории, длины.
