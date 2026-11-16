# Аудит Open-SWE-Traces v1.2 — Qwen3.8-27B: восстановимость входов и длины контекста

Датасет: [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), ревизия `4fa4c2ae9afc6b397bc50391d43450c79993c138` (2026-10-28), конфиг `v1.2`.
Аудит проведён 2026-11-15/16 по **всему** v1.2: 235 391 траектория, 13 071 706 вызовов, 30.45 GB parquet. Полный прогон шёл на многоядерном Linux-сервере проекта.
Только аудит данных: модели не запускались, speculative decoding не измерялся.

## TL;DR

- v1.2 — это **235 391 траектория** Qwen3.8-27B в 5 harness (Claude Code, OpenCode, OpenHands, SWE-agent, mini-SWE-agent), везде `enable_thinking=True` и `reasoning_effort=xhigh`.
- **Сохранено намного больше, чем в DeepSWE 1.1.** Есть system prompt, **схемы инструментов** (`tools`, у каждого harness один вариант на весь корпус), reasoning и `tool_call.id` от vLLM. **Редактирования и анонимизации нет**: тексты сырые.
- **Не сохранено:** per-call usage, `stop_reason`, sampling, `max_tokens`, версии harness, ревизия весов, token ids, logprobs. Поэтому реконструкцию нельзя сверить с сервером, и все длины здесь получены рендером через шаблон.
- Рендер через шаблон Qwen3.8-27B@`1d4bf0f2ff` **на уровне token ids согласован** с прямым рендером префикса с generation prompt: совпало 88 565 из 88 566 проверенных вызовов. Единственное расхождение — артефакт BPE на пустом reasoning, длина при этом та же.
- **Режим контекста отличается между конфигурациями.** Его выдаёт тест «стены у окна» по всем траекториям (§4):

  | Конфигурация | Траекторий | Что видела модель | Восстановимо |
  |---|---|---|---|
  | OpenHands | 25 599 | всю сохранённую историю, включая прошлый reasoning | 100% вызовов (косвенно) |
  | SWE-agent · scale-swe | 15 727 | то же | 100% (косвенно) |
  | SWE-agent · swe-rebench-v2 | 17 483 | **историю без прошлого reasoning**: в сохранённом виде 3 437 траекторий длиннее окна, до 1.37M | только в режиме «без прошлого reasoning» (гипотеза) |
  | mini-SWE-agent | 107 267 | **вероятно, без прошлого reasoning**: 522 траектории длиннее окна, без reasoning 3 | то же (гипотеза) |
  | Claude Code | 34 011 | всю историю до **auto-compact (~210K)**; сводки **выброшены**, история склеена | 98.4% вызовов (до первого сжатия) |
  | OpenCode | 35 304 | всю историю до **compaction (~130K)**; сводки выброшены | 90.0% вызовов |

- **Sampling не greedy, по крайней мере у mini.** Из 93 637 пар прогонов одной задачи с побайтно одинаковым префиксом первый ответ совпал в 59 (0.06%). У остальных harness на задачу один прогон, проверить можно только запуском модели. Сами параметры нигде не опубликованы.
- **L_max** (вход одного вызова, лучшая оценка с учётом режима контекста) по всему v1.2: median **50K**, p90 117K, p95 134K. ≥128K — 14 058 траекторий (6.0%), ≥200K — 2 109 (0.9%), ≥250K — 138. Траектории в 3–4 раза короче, чем в DeepSWE (median 186K).

## 1. Структура и объём

```
data/<harness>/qwen38_27b/{scale-swe,swe-rebench-v2}/train-XXXXX-of-YYYYY.parquet   (100 файлов)
```

Каждый файл — одна row group размером до 2 500 строк. Любая строка требует чтения всего файла, но мелкие листовые колонки (`messages.role`, имена tools, `metadata`) читаются отдельно по range-запросам. Схема одна для всех harness:

| Колонка | Содержимое |
|---|---|
| `instance_id`, `repo`, `license`, `language`, `trajectory_id`, `hf_dataset_name` | идентификаторы, источник задачи (`nebius/SWE-rebench-V2` / `AweAI-Team/Scale-SWE`) |
| `messages[]` | `role ∈ {system, user, assistant, tool}`, `content`, `reasoning_content`, `tool_calls[{id, type, function{name, arguments(JSON-строка)}}]` |
| `tools[]` | схемы инструментов, JSON-строки |
| `resolved` | 1 / 0 / −1 (unknown) |
| `metadata` | `category`, `teacher_model{name, enable_thinking, reasoning_effort}`, `reference_patch`, `model_patch` |

| Harness | Траекторий | Задач | Вызовов модели | Tool calls | Объём | Версия |
|---|---|---|---|---|---|---|
| Claude Code | 34 011 | 34 011 | 1 717 635 | 1 989 855 | 4.02 GB | ❌ |
| OpenCode | 35 304 | 35 304 | 1 791 240 | 2 087 165 | 5.72 GB | ❌ |
| OpenHands | 25 599 | 25 599 | 1 605 255 | 1 804 503 | 3.54 GB | v0.53.0 (со слов автора в обсуждении #6, для v1.0) |
| SWE-agent | 33 210 | 33 210 | 2 528 666 | 2 528 666 | 6.48 GB | ❌ |
| mini-SWE-agent | 107 267 | 41 405 (1–3 прогона) | 5 428 910 | 6 457 772 | 10.69 GB | ❌ |
| **Всего** | **235 391** | | **13 071 706** | | **30.45 GB** | |

В `data_distribution.md` указано 235 396 строк, в файлах на 5 меньше (CC −2, SWE-agent −3). Все траектории начинаются с `system, user` и заканчиваются assistant. У mini есть 12 траекторий с двумя user в начале и 370 user-сообщений в середине (ошибки формата от harness).

## 2. Полнота

| Элемент | CC | OpenCode | OpenHands | SWE-agent | mini |
|---|---|---|---|---|---|
| System prompt | ✅ 666 симв., Agent SDK + git status, свой у каждой задачи (33 447 вариантов); блоки склеены без разделителя (`…Agent SDK.CWD:`) | ✅ ~9.6K симв., 22 варианта | ✅ 8K симв., 1 вариант | ✅ 1 строка | ✅ 1 строка |
| Схемы инструментов | ✅ Bash/Read/Edit в формате Anthropic (`input_schema`) | ✅ 10 (bash, read, edit, grep, glob, write, todowrite, webfetch, task, skill) | ✅ execute_bash, str_replace_editor, think, finish | ✅ bash, str_replace_editor, submit | ✅ bash |
| Отрисованный префикс system+tools, токенов | 1 168–2 079 (median 1 255) | 7 435–8 791 | 3 938 | 1 014 | 321 |
| Reasoning | ✅ во всех ходах | ✅ | ✅ | ✅ (в swe-rebench-v2 не возвращался в контекст, §4) | ✅ (вероятно, не возвращался) |
| Tool calls: имя, аргументы | ✅ (все аргументы — валидный JSON-объект, 0 ошибок) | ✅ | ✅ | ✅ | ✅ |
| `tool_call_id` | ✅ у вызова (`chatcmpl-tool-<hex>`), у tool-сообщения ❌ → привязка по позиции | то же | то же | то же | то же |
| Tool outputs | ✅ (Read с номерами строк, как в CC) | ✅ | ✅ (`OBSERVATION:`, `<response clipped>`) | ✅ | ✅ JSON `{returncode, output}`; длинные — `output_head/tail` |
| Признак ошибки инструмента | ❌ | ❌ | ❌ | ❌ | `returncode` в тексте |
| Per-call usage, `stop_reason` | ❌ | ❌ | ❌ | ❌ | ❌ |
| Sampling, `max_tokens`, окно | ❌ | ❌ | ❌ | ❌ | ❌ |
| `reasoning_effort` / thinking | ✅ `xhigh` / `True` | ✅ | ✅ | ✅ | ✅ |
| Версия harness | ❌ | ❌ | (v0.53.0?) | ❌ | ❌ |
| Ревизия весов / шаблона | ❌ | ❌ | ❌ | ❌ | ❌ |
| Сводки compaction | ❌ **выброшены** | ❌ **выброшены** | — | — | — |
| Token ids / logprobs | ❌ | ❌ | ❌ | ❌ | ❌ |
| Редактирование / анонимизация | нет | нет | нет | нет | нет |

**Редактирование.** По всем 235 391 траекториям искались `<REDACTED_*>`, `<HOME>`, `<TMP>`. Нашлось 9 совпадений в 5 траекториях, и все они — нотация самой модели в её reasoning (`<HOME>/.cli`, `<TMP>/.fuelup/bin`). Остальные похожие строки — настоящее содержимое репозиториев (`_REDACTED_VALUE`, `<EMAIL>` в XML). E-mail в `git log` и в system prompt OpenHands не тронуты. Заметок о конверсии и выброшенных событиях в карточке нет.

**Sampling.** Параметров нет в данных, в карточке, в метаданных parquet, в сообщениях коммитов (десятки «Update README.md» / «Upload folder») и в 7 обсуждениях на HF. Статья (EMNLP 2026 Findings) описывает только v1.0. `generation_config.json` модели задаёт T=1.0, top_k=20, top_p=0.95, а харнессы по умолчанию отправляют свои значения (не проверено по коду). Проверка по данным возможна только у mini: префиксы до первого вызова у повторных прогонов одной задачи побайтно совпадают (system и user одни и те же). Из 93 637 таких пар первый ответ (reasoning + текст + tool calls) совпал в 59 (0.06%), один только reasoning — в 660 (0.7%; это короткие шаблонные фразы). `results/greedy_check.csv`. **Mini не greedy.**

**Максимальный выход одного вызова** (токены Qwen3.8, включая reasoning): CC 32.0K, mini 53.0K, OpenHands 56.0K, SWE-agent 47.7K (scale-swe) / 63.5K (swe-rebench-v2), OpenCode 131.1K. p99 по траекториям ≈ 21–23K, у SWE-agent·swe-rebench-v2 — 41K. Общего лимита 32K нет, `max_tokens` неизвестен. У CC максимум ≈32K, что похоже на `max_tokens=32000`.

## 3. Токенайзер и шаблон

- Использован `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (2026-08-14). Траектории сгенерированы в августе–сентябре 2026: в CC system prompt указано `Date: 2026-09-22`, в `ls` mini — 18 августа. Следующие коммиты модели — README (2026-11-05) и «Add a response template and generation tags» (2026-11-07, `37561816e6`), то есть после выхода v1.2. Последний меняет только теги `{% generation %}` (установлено на DeepSWE). `tokenizer.json`, `config.json` и `generation_config.json` в нём не менялись. Токенайзер на сервере скачан с HF той же ревизии, sha256 всех 5 файлов совпали с локальными.
- Свойства шаблона:
  - `preserve_thinking` по умолчанию включён: reasoning всех прошлых ходов остаётся в контексте (`<think>…</think>`);
  - `reasoning_effort` по умолчанию `xhigh` и добавляет инструкцию в начало system;
  - схемы инструментов идут блоком `# Tools … <tools>` в system;
  - tool calls рисуются как XML `<tool_call><function=…><parameter=…>`, ответы инструментов — как `<tool_response>` внутри user-хода;
  - `arguments` шаблон перебирает через `|items`; здесь, как и в vLLM, JSON-строки разбираются в dict.
- Схемы CC хранятся в формате Anthropic. Для рендера они переведены в `{"type":"function","function":{name, description, parameters}}`, как это делает OpenAI-совместимый слой vLLM. Это допущение: путь CC → сервер не задокументирован.
- `generation_config.json`: T=1.0, top_k=20, top_p=0.95. Окно 262 144.

## 4. Проверка реконструкции

Для каждого вызова j: `input_j = tokens(render(messages[:j], tools) + generation prompt)`. Траектория рендерится один раз, токенизируется с offsets, а границы ищутся по `<|im_start|>assistant\n`. Отдельно для каждой траектории считается тот же вход без прошлого reasoning (`L_max_no_past_reasoning`).

**Без usage проверка `delta_j = reported − visible` невозможна.** Вместо неё сделаны четыре внутренние проверки.

1. **Согласованность на уровне token ids.** Для 3 вызовов (первый, средний, последний) в первых 300 траекториях каждого shard сравнивались `tokens(render(messages[:j]) + gen prompt)` и отрезок полного рендера: совпало 88 565 из 88 566. Единственное расхождение (SWE-agent `de3a2f35…`, последний вызов) — артефакт BPE. Reasoning там пустой, и в полном рендере `<think>\n\n</think>` даёт токен `\n\n`, а префикс заканчивается на `<think>\n`. Строки совпадают, длина тоже. Рендер без ошибок: 235 391 из 235 391.

2. **Тест «стены у окна».** Если модель видела именно то, что считается, вход не может превысить 262 144, а траектории, упёршиеся в окно, накапливаются прямо под ним. Распределение L_max по корзинам (тыс. токенов), по всем траекториям:

   | Конфигурация | режим | 230–245 | 245–255 | 255–262 | 262–280 | 280–350 | >350 |
   |---|---|---|---|---|---|---|---|
   | OpenHands (оба источника) | с reasoning | 160 | 73 | 34 | **0** | **0** | **0** |
   | SWE-agent · scale-swe | с reasoning | 82 | 53 | 26 | **0** | **0** | **0** |
   | SWE-agent · swe-rebench-v2 | с reasoning | 484 | 279 | 203 | 421 | 1 158 | 1 858 |
   | SWE-agent · swe-rebench-v2 | **без** reasoning | 17 | 4 | 2 | **0** | **0** | **0** |
   | mini (оба источника) | с reasoning | 268 | 134 | 85 | 170 | 286 | 66 |
   | mini (оба источника) | **без** reasoning | 9 | 5 | 0 | 1 | 1 | 1 |
   | Claude Code | с reasoning | 84 | 31 | 20 | 64 | 92 | 79 |
   | OpenCode | с reasoning | 228 | 97 | 45 | 87 | 201 | 106 |

   Выводы:
   - **OpenHands и SWE-agent·scale-swe:** обрыв у окна при учёте reasoning. Прошлый reasoning был в контексте, то есть режим шаблона по умолчанию.
   - **SWE-agent·swe-rebench-v2:** с reasoning 3 437 траекторий длиннее окна (до 1.37M), без reasoning ни одной. Прошлый reasoning **не** возвращался. Это другая конфигурация запуска того же harness. Её shards тяжелее в 2.5 раза на строку, а длинные траектории показывают резкие скачки: reasoning по 10K+ токенов на ход (`figures/context_over_session.png`).
   - **mini:** то же, но с 3 исключениями (без reasoning всё равно > окна, до 361K). Режим «без прошлого reasoning» вероятен, но не доказан: стены нет ни в одном варианте. Траектория `b3f5c209…` дошла в сохранённом виде до 303K и штатно завершилась `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   - **CC и OpenCode:** даже без reasoning 18 и 42 траектории длиннее окна, то есть контекст сжимался (п. 4).

3. **Тест на greedy** (§2): mini не greedy.

4. **Скрытое сжатие у CC и OpenCode.** Модель ссылается на сводку, которой в данных нет. Пример из CC (`b9c17fed…`, вызов ~100, видимый вход ≈220K): *«From the summary: "visit_end pops ctrl, special-cases If…"»*; другой — *«In the previous (pre-compaction) run, did this pass? The summary said…»*. Сообщений «This session is being continued…» нет, падения длины тоже нет: история просто склеена. Порог оценён по входу в момент **первой строгой ссылки** на сводку (`results/strict_summary_ref_hist.csv`):

   | | …120–125 | 125–130 | **130–135** | 135–140 | … | 190–200 | 200–210 | **210–230** | 230–262 | >262 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | OpenCode | 13 | 14 | **579** | 460 | … | 48 | 31 | 36 | 60 | 35 |
   | Claude Code | 10 | 9 | 11 | 6 | … | 2 | 0 | **47** | 233 | 25 |
   | OpenHands (фон) | 19 | 11 | 14 | 13 | … | 4 | 12 | 3 | 1 | 0 |

   Ссылки до порога — фон ложных срабатываний того же уровня, что у OpenHands («отмечу в summary», «summary table»). Принятые пороги: **OpenCode 130K, CC 210K**. Вызовы после первого пересечения порога считаются невосстановимыми (сам вызов на пороге ещё идёт до сжатия).

**Скрытый префикс для одной задачи в разных прогонах.** У mini system prompt один на корпус, а префикс до первого вызова у повторных прогонов совпадает побайтно (на этом построен тест на greedy). У CC system prompt содержит git status, так что он свой у каждой задачи. У OpenCode 22 варианта system prompt.

**Прямой ответ.**
- **Доказать точное восстановление `input_ids` по данным нельзя ни для одного harness**: нет серверных счётчиков.
- **OpenHands и SWE-agent·scale-swe** (41.3K траекторий, 2.59 M вызовов): все косвенные проверки согласуются, вход — рендер сохранённых сообщений с прошлым reasoning. Это лучший кандидат для replay.
- **SWE-agent·swe-rebench-v2 и mini** (124.8K траекторий): вход, вероятно, — рендер без прошлого reasoning (пустые `<think></think>` у прошлых ходов). Для SWE-agent·rebench это подтверждено стеной у окна, для mini — только отсутствием переполнения. Точный формат «без reasoning» неизвестен: пустой `<think>` или его отсутствие. Это ±несколько токенов на ход.
- **CC и OpenCode**: вход восстанавливается только **до первого сжатия**: 98.4% и 90.0% вызовов, 616 (1.8%) и 4 332 (12.3%) траекторий затронуты. После сжатия не хватает текста сводки.
- Оговорки для всех:
  - неизвестно, передавался ли `reasoning_effort` (шаблон по умолчанию всё равно ставит `xhigh`);
  - точный путь конверсии запросов CC и OpenCode неизвестен;
  - не исключено, что harness вставлял что-то в запрос мимо лога: у CC нет обычных `<system-reminder>`, а промпт Agent SDK всего ~1.3K токенов.
- **Output token ids** не сохранены. Для экспериментов с acceptance они не нужны: продолжения генерируются заново.
- **On-policy воспроизводимость не подтверждена.** Нет ревизии весов, sampling (mini точно не greedy, остальные неизвестны), `max_tokens`, версий harness.

## 5. Ловушки usage

| Проверка | Результат |
|---|---|
| Σ per-call = итоговым суммам | неприменимо: usage нет ни по вызовам, ни итоговых |
| Семантика «peak context» | в датасете такой метрики нет. Здесь L_max = max(input), выход — отдельной колонкой |
| Шаги со слитым usage | неприменимо |
| Неудачные вызовы / вырожденные выводы | **отфильтрованы при конверсии** (§5.1). Исключение — 4 обрыва OpenCode по `max_tokens`: они сохранены, но это всегда последнее сообщение траектории |
| Цензурирование окном / порогом | OpenHands и SWE-agent·scale-swe — окно 262 144 (0 выше). CC и OpenCode — порог сжатия (L_max ≤ 226K и ≤ 172K в восстановимой части; на гистограмме видна «стопка» у порога). SWE-agent·rebench — окно в режиме без reasoning (max 259.9K). mini — неясно |

### 5.1. Аномалии (разобраны по содержимому сообщений)

Скрипты `inspect_anomalies.py`, `midturn_check.py`, `escape_check.py`.

- **Выброшенные неудачные ответы (mini).**
  - Есть 370 сообщений harness `Tool call error. Every response needs to use the 'bash' tool…` в 352 траекториях. Это штатная реакция mini на ответ без tool call.
  - 356 из них идут сразу после tool-сообщения, 14 — после такого же сообщения об ошибке, после assistant — **ни одного**. Значит, сам ответ модели без tool call удалён при конверсии, а ответ harness оставлен.
  - Реальный контекст следующих вызовов в этих траекториях длиннее сохранённого на 1–2 хода модели.
  - У других harness таких сообщений нет. Совпадения регулярки в tool-сообщениях — обычный вывод программ. Либо ошибок формата не было, либо удалено и то, и другое.
- **Обрыв по `max_tokens` (OpenCode).** 4 вызова (из 1.79 M) имеют выход ровно 131 076 токенов: это 131 072 токена reasoning плюс 4 токена, которые добавляет шаблон (`\n`, `</think>`, `\n\n`, `<|im_end|>`). `content` пустой, tool calls нет, reasoning заканчивается вырожденным циклом (`(2/2)·(2/2)·…`). Все 4 — последнее сообщение траектории, так что на реконструкцию последующих вызовов не влияют. Вывод: у OpenCode `max_tokens = 131072`. У CC максимум 31 960 при 1.7 M вызовов, что похоже на `max_tokens = 32000`. У остальных скопления у лимита нет.
- **Лимит наблюдения SWE-agent.** Выводы около 100K токенов заканчиваются штатной пометкой `<response clipped><NOTE>Observations should not exceeded 100000 characters. N characters were elided…`. Лимит — 100 000 символов. Длинные ряды чисел дают около 1 токена на символ (Qwen токенизирует цифры по одной). Модель видела этот текст как есть.
- **JSON-экранирование в выводах mini.** Выводы хранятся как JSON `{returncode, output|output_head/output_tail, elided_chars, warning}` с экранированием `\uXXXX`: кириллица, а также `'`, `<`, `>`, `&`. Экранирование есть в 53% tool-ответов и добавляет в них +18% токенов. Mini усекает по символам исходного текста, поэтому на кириллице сохранённый вывод бывает в разы длиннее порога: 36.9K токенов у faker/ru_RU. В DeepSWE 1.1 mini хранит выводы так же (52%, +14%), и там реконструкция с экранированием совпадала с сервером в пределах 0.5% у 92% вызовов. Значит, модель видит именно экранированный JSON, и реконструкция здесь верна.
- **Завершённость траекторий** (`termination.py`, `results/termination_by_config.csv`). 235 349 из 235 391 (99.98%) заканчиваются штатным действием агента: финальный ответ (CC, OpenCode), `finish` (OpenHands), `submit` (SWE-agent), `COMPLETE_TASK…` (mini). Не завершены только 42: 10 у OpenCode с пустым финальным ходом (из них 4 обрыва по 131K) и 32 у SWE-agent на обычном tool call. Видимо, авторы оставили только завершённые траектории. Повторяемость в последнем ходе у завершённых почти нулевая (доля повторяющихся 20-символьных фрагментов в хвосте: p99 = 5%). **Длинные траектории — завершённые, но в основном неуспешные**: среди L_max ≥ 128K решено 24% (16–35% по конфигурациям) против ~50% по корпусу. Для выборки под speculative decoding достаточно выкинуть эти 42. Фильтр по `resolved` почти убрал бы длинный контекст, поэтому лучше стратифицировать по `resolved` и длине.
- **3 траектории mini длиннее окна даже без reasoning** (`c3204b01…`, `2c11edfa…`, `61f0b570…`, все scale-swe, `resolved=1`, до 361K). Выводы инструментов в них в сумме дают 196–259K токенов. Экранирование их не объясняет, так как оно реально. Возможные причины: другое окно сервера в части прогонов mini или склейка нескольких запусков. По данным не различить.

## 6. Статистика L_max (весь v1.2)

`L_{i,j}` — вход **одного** вызова (рендер через шаблон Qwen3.8 + generation prompt). `L_i^max = max_j`.
- **Лучшая оценка** (`L_max_best_estimate`): для `preserve`-конфигураций — с прошлым reasoning, только вызовы до первого сжатия (CC, OpenCode); для `no_past_reasoning` — без прошлого reasoning.
- **Как сохранено** (`L_max_visible_template`): все сообщения с reasoning. Для сравнения.
- Сумма по сессии — отдельной колонкой (`session_prompt_tokens_sum`). Она в 24–43 раза больше L_max (медиана), см. `figures/session_sum_vs_lmax.png`.

**Лучшая оценка:**

| Конфигурация (Qwen3.8-27B · xhigh) | режим | n | p25 | median | p75 | p90 | p95 | max | ≥32K | ≥64K | ≥100K | ≥128K | ≥200K | ≥250K | ≥256K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Claude Code | до сжатия | 34 011 | 37.4K | 60.8K | 94.6K | 136.8K | 166.1K | 226.5K | 27 311 | 15 475 | 7 612 | 3 876 | 780 | 0 | 0 |
| OpenCode | до сжатия | 35 304 | 40.5K | 65.5K | 100.7K | 130.2K | 131.0K | 171.8K | 29 352 | 17 622 | 8 980 | 1 717 | 0 | 0 | 0 |
| OpenHands | с reasoning | 25 599 | 54.5K | 78.0K | 112.2K | 152.3K | 180.2K | 261.8K | 24 391 | 16 018 | 8 330 | 4 213 | 746 | 72 | 0 |
| SWE-agent · scale-swe | с reasoning | 15 727 | 42.4K | 64.6K | 99.1K | 143.3K | 174.2K | 261.0K | 13 574 | 7 723 | 3 852 | 2 039 | 418 | 56 | 0 |
| SWE-agent · swe-rebench-v2 | без reasoning | 17 483 | 42.4K | 61.0K | 86.0K | 114.9K | 134.6K | 259.9K | 15 334 | 7 738 | 2 843 | 1 001 | 89 | 4 | 0 |
| mini-SWE-agent | без reasoning | 107 267 | 25.3K | 37.2K | 54.9K | 77.4K | 94.3K | 361.2K | 63 109 | 17 360 | 4 233 | 1 212 | 76 | 6 | 3 |
| **Все v1.2** | смешанный | **235 391** | 31.8K | **50.2K** | 79.4K | 117.3K | 134.4K | 361.2K | 173 071 | 81 936 | 35 850 | 14 058 | 2 109 | 138 | 3 |

**Как сохранено (с reasoning, без отсечения сжатия)** — для сравнения:

| Конфигурация | median | p90 | p95 | max | ≥128K | ≥200K | ≥256K |
|---|---|---|---|---|---|---|---|
| Claude Code | 60.8K | 136.8K | 166.1K | 847.4K | 3 876 | 780 | 235 |
| OpenCode | 65.5K | 137.9K | 181.7K | 823.7K | 4 216 | 1 315 | 394 |
| SWE-agent · swe-rebench-v2 | 139.1K | 358.4K | 466.6K | 1 374.7K | 9 258 | 5 511 | 3 437 |
| mini-SWE-agent | 57.6K | 129.2K | 158.9K | 392.2K | 10 258 | 2 024 | 522 |
| Все v1.2 | 65.1K | 151.1K | 194.3K | 1 374.7K | 33 860 | 10 794 | 4 588 |

Пороги: 32K=32 768, 64K=65 536, 100K=100 000, 128K=131 072, 200K=200 000, 250K=250 000, 256K=262 144. Порог 256K совпадает с окном и строго достичь его нельзя. Три траектории mini ≥ окна в «лучшей оценке» — аномалии из §4.2.

Прочее (`results/validation_by_config.csv`):
- медиана по траекториям медианного размера tool output — 148–294 токена;
- p95 максимального вывода — 5.7K (mini), 8.7K (OpenHands), 14.9K / 21.1K (SWE-agent scale / rebench), 18.1K (CC), 18.4K (OpenCode).

## 7. Tool calls и compaction

- **Границы и типы tool calls** восстанавливаются полностью: имя, аргументы, порядок. У вызова есть id, а ответ привязывается по позиции: N tool-сообщений подряд после assistant с N вызовами. Последний вызов (`submit` / `finish` / `COMPLETE_TASK…`) ответа не имеет. Параллельные вызовы есть у всех, кроме SWE-agent (1.12–1.19 tool call на ход).
- **Распределение инструментов** (`results/tools_full_corpus.csv`):

  | Harness | Инструменты |
  |---|---|
  | Claude Code | Bash 75.5%, Read 15.1%, Edit 9.4% |
  | OpenCode | bash 59.7%, read 21.4%, edit 10.6%, grep 3.8%, write 2.4%, todowrite 1.2%, glob 0.9%; webfetch, task, skill — единицы |
  | OpenHands | execute_bash 74.0%, str_replace_editor 24.4%, finish 1.4%, think 0.2% |
  | SWE-agent | bash 77.2%, str_replace_editor 20.2%, submit 2.7% |
  | mini-SWE-agent | bash 100% |

- **Ошибки инструментов** явного флага не имеют. У mini есть `returncode` в JSON, у остальных ошибка видна только в тексте.
- **Compaction.** CC и OpenCode сжимали контекст. Явных маркеров нет, текст сводки и вызов-суммаризатор **не сохранены**, а сообщения до и после склеены. Обнаружить сжатие можно по видимой длине выше порога и по ссылкам модели на сводку. Пороги около 210K (CC) и 130K (OpenCode). Длину контекста после сжатия восстановить нельзя.
  - Затронуто: CC 616 траекторий (1.8%), 27 005 вызовов (1.6%); OpenCode 4 332 (12.3%), 178 950 вызовов (10.0%).
  - Субагенты: у OpenCode `task` есть в схеме, но вызывается единицы раз, и вызовы субагентов в траекторию не попали. У CC инструмента `Agent`/`Task` нет.
  - OpenHands, SWE-agent и mini сжатия не показывают: маркеры condenser и `Old environment output` не найдены. У SWE-agent·rebench и mini «сжатием» служит, по-видимому, выбрасывание прошлого reasoning.

## 8. Сравнение с DeepSWE 1.1 (`experiments/deepswe11-audit/`)

| | DeepSWE 1.1 | Open-SWE-Traces v1.2 |
|---|---|---|
| Масштаб Qwen3.8 | 565 траекторий (113 задач × 5 прогонов) | 235 391 траектория, 13.1 M вызовов |
| Harness | Mini-SWE, Claude Code, Pi | Mini-SWE, Claude Code, **OpenCode, OpenHands, SWE-agent**; Pi нет |
| Effort | low / medium / xhigh | только xhigh |
| System prompt | только Mini-SWE | ✅ все |
| Схемы инструментов | ❌ | ✅ все |
| Редактирование текста | всё (~190K замен) | нет |
| Per-call usage | ✅ серверный | ❌ |
| Sampling | ✅ в `run.json` | ❌ |
| Сводки compaction | CC ✅, Pi ❌ | CC ❌, OpenCode ❌ |
| Возврат прошлого reasoning | да (подтверждено сервером) | да / нет в зависимости от конфигурации (выведено косвенно) |
| Проверка реконструкции | против сервера: Pi и Mini 92% в пределах 0.5% | внутренняя: token ids 88 565/88 566 + тест стены у окна |
| L_max median | 186K | 50K (лучшая оценка) |
| ≥128K | 81% | 6% |

Общие harness — Mini-SWE и Claude Code, но конфигурации разные:
- у CC в DeepSWE было ~23 инструмента и скрытый префикс ~28K токенов, здесь — Agent SDK с 3 инструментами и префиксом ~1.3K;
- у mini в DeepSWE reasoning возвращался в контекст (сервер это подтвердил), здесь, вероятно, нет.

**Для replay и измерения speculative decoding** Open-SWE-Traces v1.2 подходит **лучше** по главному критерию: входные префиксы можно построить из данных (system prompt, схемы, сырые тексты). Лучшие подмножества:
- **OpenHands + SWE-agent·scale-swe**: 41.3K траекторий, 2.59 M вызовов, режим контекста согласован с окном;
- **CC и OpenCode до первого сжатия**: ещё 3.3 M вызовов.

Хуже DeepSWE этот датасет в трёх отношениях:
- длины нельзя сверить с сервером;
- режим контекста приходится выводить;
- контексты в 3–4 раза короче.

Для изучения длинного контекста (≥128K) DeepSWE богаче, для массового replay — Open-SWE-Traces.

## 9. Файлы

`results/` (в этом каталоге):
- `trajectories.csv.gz` — 235 391 строка, по одной на траекторию. Главные колонки:
  - `config`, `context_mode`;
  - `L_max_best_estimate`, `L_max_visible_template`, `L_max_no_past_reasoning`, `L_max_preserve_reconstructible`, `L_max_call_index`;
  - `prefix_tokens_system_tools`, `session_prompt_tokens_sum`, `output_tokens_max`;
  - `n_calls_over_window`, `over_window_*`, `suspected_hidden_compaction`, `first_call_over_thr`;
  - `tool_calls_by_type`, `tool_output_tokens_*`, `n_redaction_placeholders`, `exact_checked/exact_ok`;
  - `file/row` — адрес в shard.
- `summary_by_config.csv` — таблицы из §6 (все метрики, включая сумму по сессии).
- `validation_by_config.csv` — проверки из §2–§4.
- `greedy_check.csv`, `strict_summary_ref_hist.csv`, `tools_full_corpus.csv`.
- `figures/lmax_ecdf.png`, `lmax_hist_by_config.png`, `context_over_session.png`, `session_sum_vs_lmax.png`.
- `aux/`: `parquet_meta.json` (footer всех 100 shards), `tools_variants.json` (схемы инструментов), `scan_strings.json` (поиск плейсхолдеров и маркеров с примерами, весь корпус).

Крупные файлы в `data/open-swe-traces/` (исключён из Git):
- `audit/final/calls.csv.gz` — 13 071 706 строк, по одной на вызов: `input_visible_template`, `output_template`, `reasoning_chars`, `content_chars`, `n_tool_calls`, `tool_names`, `config`, `context_mode`, `after_suspected_compaction`; 186 MB.
- `audit/final/trajectories.csv` — несжатая копия таблицы траекторий.
- `audit/full/` — сырые выходы прогона (`signals.csv` — ссылки на сводку, маркеры усечения и хеши для greedy по каждой траектории, `reasoning_share.csv`).
- `light/light.parquet` — лёгкий обзор всех траекторий по мелким колонкам.
- `sample/sample_8_per_harness.parquet` — по 8 траекторий из подвыборки: min, p10, p25, p50, p75, p90, p99, max L_max.
- `tokenizer/qwen3.8-27b@1d4bf0f2ff/`, `meta/` (README, `data_distribution.md`, дерево файлов).

Полный корпус (30.45 GB) лежит на сервере проекта в `~/specdec-open-swe/data/open-swe-traces/shards/`. Локально — только 5 shards подвыборки (`data/open-swe-traces/shards/`).

## 10. Воспроизведение

Метаданные и лёгкий обзор (macOS, `UV_PROJECT_ENVIRONMENT=.venv-mac`, `HF_TOKEN` в `.env`; скрипты запускаются через `python -I` и лежат отдельно от данных):

```bash
uv sync    # pyarrow добавлен в pyproject
source experiments/open-swe-traces-audit/hfenv.sh     # HF_TOKEN (только эта строка .env), D, X, PY (zsh-массив), REV
$PY $X/list_tree.py nvidia/Open-SWE-Traces $REV $D/meta/tree.json            # дерево файлов (постранично)
$PY $X/parquet_meta.py $D/meta/tree.json $REV $D/meta/parquet_meta.json      # footer'ы v1.2, range-запросы
$PY $X/survey_light.py $D/meta/parquet_meta.json $REV $D/light               # мелкие колонки всех 235K, ~6 мин
```

Полный прогон (Linux, много ядер; на 80 ядрах: загрузка 47 мин, обработка ~1 ч 45 мин):

```bash
mkdir -p ~/specdec-open-swe/{scripts,logs,data/open-swe-traces/tokenizer} && cd ~/specdec-open-swe
python3 -m venv .venv && .venv/bin/pip install transformers==5.23.0 tokenizers==0.23.3 huggingface_hub==2.4.0 \
  pyarrow==26.0.1 pandas==3.1.1 numpy==2.5.4 jinja2==3.1.6 matplotlib==3.11.2 hf_xet==1.8.0
# скопировать сюда scripts/*.py и run_full.sh; HF_TOKEN передавать через stdin/окружение, не на диск
.venv/bin/hf download nvidia/Open-SWE-Traces --repo-type dataset --revision 4fa4c2ae9afc6b397bc50391d43450c79993c138 \
  --include "data/*/qwen38_27b/*/*.parquet" --local-dir data/open-swe-traces/shards --max-workers 16
.venv/bin/hf download Qwen/Qwen3.8-27B chat_template.jinja tokenizer.json tokenizer_config.json generation_config.json config.json \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir data/open-swe-traces/tokenizer/qwen3.8-27b@1d4bf0f2ff
WORKERS=76 nohup bash scripts/run_full.sh > logs/run_full.log 2>&1 &   # audit → reasoning_share → signals → scan_strings
```

Сводка и графики (локально, после копирования `audit/full/` с сервера):

```bash
$PY $X/summarize.py $D/audit/full $D/light/light.parquet $X/results
$PY $X/plots.py $X/results $X/results/figures
```

Скрипты:
- `audit.py` — рендер, токенизация, длины по вызовам, проверка token ids;
- `reasoning_share.py` — длины без прошлого reasoning;
- `signals.py` — ссылки на сводку, маркеры усечения, хеши для greedy;
- `scan_strings.py` — плейсхолдеры и маркеры;
- `summarize.py` — режимы контекста, пороги, таблицы;
- `plots.py`;
- `make_sample.py` — выборка по 8 траекторий;
- `check_exact.py` — разбор одной траектории: расхождение token ids и контекст плейсхолдеров;
- `inspect_anomalies.py`, `midturn_check.py`, `escape_check.py` — разбор аномалий (§5.1);
- `peek.py` — просмотр сообщений.

## Открытые вопросы

1. Проверить запуском модели, greedy ли OpenHands, SWE-agent, CC и OpenCode. Mini точно нет.
2. Узнать у авторов (обсуждение на HF): `max_tokens`, sampling, окно сервера, версии harness, путь конверсии CC и OpenCode, почему в SWE-agent·swe-rebench-v2 и mini не возвращался reasoning, как выбрасывались сводки compaction и неудачные вызовы.
3. Если для экспериментов нужны вызовы после сжатия у CC и OpenCode, их придётся пересобирать: регенерировать сводку той же моделью. Это уже не реплей исходного контекста.
