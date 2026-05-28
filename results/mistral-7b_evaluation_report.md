# Safety Evaluation Report: MISTRAL-7B

This report evaluates defenses on target model **mistralai/Mistral-7B-Instruct-v0.3** grounded in the TRiSM framework.

## 1. Summary Metrics

| Configuration | ASR (Attack Success Rate) | DSR (Defense Success Rate) | FPR (False Positive Rate) |
| --- | --- | --- | --- |
| Baseline | 68.00% | 32.00% | 0.00% |
| D1 Only | 68.00% | 32.00% | 0.00% |
| D2 Only | 48.00% | 52.00% | 3.85% |
| D3 Only | 32.00% | 68.00% | 0.00% |
| D1 + D2 | 48.00% | 52.00% | 3.85% |
| D2 + D3 | 32.00% | 68.00% | 3.85% |
| D1 + D3 | 32.00% | 68.00% | 0.00% |
| All Three | 32.00% | 68.00% | 3.85% |

## 2. Per-Layer Defense Breakdown

| Configuration | Caught by D1 (Input Filter) | Caught by D2 (Alignment) | Caught by D3 (Output Filter) | Bypassed (Jailbroken) |
| --- | --- | --- | --- | --- |
| Baseline | 0 | 0 | 0 | 17 |
| D1 Only | 1 | 0 | 0 | 17 |
| D2 Only | 0 | 13 | 0 | 12 |
| D3 Only | 0 | 0 | 11 | 8 |
| D1 + D2 | 1 | 12 | 0 | 12 |
| D2 + D3 | 0 | 11 | 6 | 8 |
| D1 + D3 | 1 | 0 | 11 | 8 |
| All Three | 1 | 10 | 6 | 8 |

## 3. Findings & Observations

- **ASR (Baseline)** represents the model's vulnerability before applying safety defenses.
- **D1 Only** catches simple keyword matching, role-plays, and special characters.
- **D2 Only** hardens the system prompt to explicitly enforce refusal guidelines.
- **D3 Only** relies on LLaMA-Guard-3-8B classification on the model response.
- **All Three** combine all defenses for a layered defense-in-depth framework.
