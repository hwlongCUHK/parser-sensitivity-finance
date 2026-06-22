# Parsing the Score: How Answer Extraction Changes Who Wins

Code and data for the ICAIF 2026 paper on extraction sensitivity in financial LLM benchmarks.

## Structure

```
release/
├── code/              # Experiment scripts (model inference)
│   ├── p4_multiparser_eval.py    # 5-parser evaluation on FinQA/FinanceReasoning
│   ├── p5_gsm8k_comparison.py    # GSM8K parser comparison
│   └── p6_multi_benchmark.py     # Multi-benchmark evaluation (SVAMP/ASDiv/MultiArith/FLARE/DROP)
├── analysis/          # Analysis scripts (CPU-only, reads existing data)
│   ├── analysis_oracle.py         # Oracle upper bound
│   ├── analysis_bootstrap.py      # Bootstrap confidence intervals
│   ├── analysis_agreement.py      # Cross-parser agreement matrix
│   ├── analysis_ensembles.py      # Best 2/3-parser combinations
│   ├── analysis_rankings.py       # Kendall τ / rank reversal analysis
│   ├── analysis_counterfactual.py # Format-stripping counterfactual
│   └── analysis_real_frameworks.py # Real framework extraction (lm-eval/FinQA/PIXIU)
├── data/              # All parser results (JSON)
│   ├── p4_finqa_*_parsers.json    # FinQA parser results (3 models)
│   ├── p5_gsm8k_*_parsers.json    # GSM8K parser results
│   ├── p6_*_parsers.json          # SVAMP/ASDiv/MultiArith/FLARE results
│   └── analysis_*.json            # Analysis outputs
└── README.md
```

## Models

- Qwen3-1.7B, Qwen3-8B (open-weight, non-thinking mode)
- GLM-4-9B-Chat (cross-family validation)

## Parsers

Five parsers tested on identical model outputs:
1. `eq_number` — first number after `=` sign
2. `boxed` — content inside `\boxed{}`
3. `fixed` — boxed-first, =NUMBER fallback
4. `last_number` — last numeric token
5. `finance_aware` — fixed + currency/%/M/B/K/sign handling

## Benchmarks

- FinQA (96 problems), FLARE-FinQA (100)
- GSM8K, SVAMP, ASDiv, MultiArith (100 each)

## Reproduction

```bash
pip install torch transformers tqdm numpy

# Run multi-parser evaluation
python code/p6_multi_benchmark.py <model> <dataset> [max_tokens] [gpu_id]

# Run analysis (CPU-only, reads results_icaif/*.json)
python analysis/analysis_bootstrap.py
```

## Key Results

- Res-B (format-compatible parsers): 2-20pp on finance vs 1-7pp on math
- Oracle ceiling: 1-2pp improvement from perfect multi-parser ensemble
- Kendall τ = -1.00 between extreme parsers on FinQA
- Real-framework spread: 11.5pp on identical FinQA outputs
