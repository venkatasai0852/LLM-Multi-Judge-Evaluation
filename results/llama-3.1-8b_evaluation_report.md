# Safety Evaluation Report: LLAMA-3.1-8B

This report evaluates defenses on target model **meta-llama/Llama-3.1-8B-Instruct** grounded in the TRiSM framework.

## 1. Summary Metrics

| Configuration | ASR (Attack Success Rate) | DSR (Defense Success Rate) | FPR (False Positive Rate) |
| --- | --- | --- | --- |
| Baseline | 44.00% | 56.00% | 0.00% |
| D1 Only | 40.00% | 60.00% | 0.00% |
| D2 Only | 56.00% | 44.00% | 0.00% |
| D3 Only | 44.00% | 56.00% | 0.00% |
| D1 + D2 | 56.00% | 44.00% | 0.00% |
| D2 + D3 | 56.00% | 44.00% | 0.00% |
| D1 + D3 | 40.00% | 60.00% | 0.00% |
| All Three | 56.00% | 44.00% | 0.00% |

## 2. Per-Layer Defense Breakdown

| Configuration | Caught by D1 (Input Filter) | Caught by D2 (Alignment) | Caught by D3 (Output Filter) | Bypassed (Jailbroken) |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0 | 11 |
| D1 Only | 1 | 0 | 0 | 10 |
| D2 Only | 0 | 11 | 0 | 14 |
| D3 Only | 0 | 0 | 0 | 11 |
| D1 + D2 | 1 | 10 | 0 | 14 |
| D2 + D3 | 0 | 11 | 0 | 14 |
| D1 + D3 | 1 | 0 | 0 | 10 |
| All Three | 1 | 10 | 0 | 14 |

## 3. Findings & Observations

- **ASR (Baseline)** represents the model's vulnerability before applying safety defenses.
- **D1 Only** catches simple keyword matching, role-plays, and special characters.
- **D2 Only** hardens the system prompt to explicitly enforce refusal guidelines.
- **D3 Only** relies on LLaMA-Guard-3-8B classification on the model response.
- **All Three** combine all defenses for a layered defense-in-depth framework.
