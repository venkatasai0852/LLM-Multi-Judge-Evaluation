# Safety Evaluation Report: QWEN-2.5-7B

This report evaluates defenses on target model **Qwen/Qwen2.5-7B-Instruct** grounded in the TRiSM framework.

## 1. Summary Metrics

| Configuration | ASR (Attack Success Rate) | DSR (Defense Success Rate) | FPR (False Positive Rate) |
| --- | --- | --- | --- |
| Baseline | 52.00% | 48.00% | 0.00% |
| D1 Only | 48.00% | 52.00% | 0.00% |
| D2 Only | 36.00% | 64.00% | 0.00% |
| D3 Only | 40.00% | 60.00% | 0.00% |
| D1 + D2 | 32.00% | 68.00% | 0.00% |
| D2 + D3 | 32.00% | 68.00% | 0.00% |
| D1 + D3 | 36.00% | 64.00% | 0.00% |
| All Three | 28.00% | 72.00% | 0.00% |

## 2. Per-Layer Defense Breakdown

| Configuration | Caught by D1 (Input Filter) | Caught by D2 (Alignment) | Caught by D3 (Output Filter) | Bypassed (Jailbroken) |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0 | 13 |
| D1 Only | 1 | 0 | 0 | 12 |
| D2 Only | 0 | 16 | 0 | 9 |
| D3 Only | 0 | 0 | 5 | 10 |
| D1 + D2 | 1 | 16 | 0 | 8 |
| D2 + D3 | 0 | 16 | 1 | 8 |
| D1 + D3 | 1 | 0 | 5 | 9 |
| All Three | 1 | 16 | 1 | 7 |

## 3. Findings & Observations

- **ASR (Baseline)** represents the model's vulnerability before applying safety defenses.
- **D1 Only** catches simple keyword matching, role-plays, and special characters.
- **D2 Only** hardens the system prompt to explicitly enforce refusal guidelines.
- **D3 Only** relies on LLaMA-Guard-3-8B classification on the model response.
- **All Three** combine all defenses for a layered defense-in-depth framework.
