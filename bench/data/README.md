# Evaluation data (third-party, unmodified)

| File | Source | Split | Rows | Licence |
|---|---|---|---|---|
| gsm8k.parquet | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (main) | test | 1,319 | MIT |
| humaneval.parquet | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | test | 164 | MIT |
| mmlu_pro.parquet | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | test | 12,032 | MIT |

They are copied here so every run uses the same rows. `quality.py mmlu_pro` samples 20 questions per category
(280 in all) with `random_state=0`. `swarm_bench.py`, `longctx.py` and `prof_longctx.py` use the MMLU-Pro question text
only as filler for long prompts.
