# Qwen3.8-27B: acceptance драфтеров на длинных контекстах

Код к отчёту [`reports/issue11-20261111-qwen3.8-27b-drafters`](../../reports/issue11-20261111-qwen3.8-27b-drafters/README.md).

| Ключ | Чекпойнт |
|---|---|
| target | `Qwen/Qwen3.8-27B` |
| `mtp` | встроенный MTP target |
| `dflash2` | `z-lab/Qwen3.8-27B-DFlash2` |
| `dspark_radixark` | `RadixArk/Qwen3.8-27B-DSpark` |
| `dspark_redhat` | `RedHatAI/Qwen3.8-27B-speculator.dspark` |
| `eagle4` | `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (нужен vLLM-форк авторов `eagle_harness` и `VLLM_USE_V2_MODEL_RUNNER=1`) |

Ревизии всех чекпойнтов — в `configs/*.json`.

## Данные

`prepare_data.py` собирает данные в `data/prompts/`.

- Короткие (до 1K токенов): из issue #11 первые 20 MATH-500, HumanEval, UltraChat и BBH; AIME 2026,
  HMMT Feb 2026, LiveCodeBench (20 hard), WildChat (20).
- Длинные: issue #10 (arXiv, GovReport, Multi-LexSum, NarrativeQA, MRCR); SWE-bench-Live с oracle-входом,
  по 10 на диапазон 4–128K; MRCR и LongBench v2 на 128–250K.

`data/issue10/` в репозиторий не входит: это файлы из issue #10 вида `<domain>/<bin>.json`, их хеши
лежат в `data/issue10.sha256`. Добавленные нами наборы — на
[Google Drive](https://drive.google.com/drive/folders/1V-Q5TH3UqxKF81V3gPvUdy-KPdk01hNO).

## Запуск

```bash
python prepare_data.py
python launch.py --tp 2 --methods ar mtp dflash2 dspark_radixark dspark_redhat eagle4 -- \
  --config configs/natural.json --prompts data/prompts/natural.jsonl --output results/natural/{method}
python collect.py && python analyze.py && python report.py && python viewer.py && python examples.py
```

`launch.py` делит видимые GPU на группы по `--tp` и раздаёт им шарды. Повторный запуск докачивает
недостающее. noeos — то же самое с `configs/noeos.json` и `data/prompts/noeos.jsonl`.

Teacher forcing (forced):

```bash
python prepare_ref.py
python launch.py --tp 2 --methods ar -- --config configs/ref.json \
  --prompts data/prompts/ref.jsonl --output results/ref/ar
python prepare_forced.py
python launch.py --tp 2 --methods mtp dflash2 dspark_radixark dspark_redhat eagle4 -- \
  --config configs/ref.json --prompts data/prompts/forced.jsonl --output results/forced/{method} \
  --ref results/ref/ar --kv-fill 0.92 --max-num-seqs 48
```

1. Эталон — жадная траектория AR с запрещённым EOS.
2. `prepare_forced.py` обрезает его на старте первого цикла + 4096 токенов.
3. Плагин vLLM `specbench/force.py` подменяет logits target на one-hot по эталонному токену. Запрос,
   который не воспроизвёл эталон, не учитывается.

## Файлы

- `specbench/run.py` — прогон одного метода на шарде промптов.
- `specbench/trace.py` — accepted/proposed и длина контекста на каждом раунде.
- `collect.py` → `tables/`, `analyze.py` → `report/data.json`, `report.py` → `report/index.html`,
  `examples.py` + `viewer.py` → просмотрщик примеров.
