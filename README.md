# LLM Jailbreak Defense Evaluation Pipeline

A research pipeline for evaluating open-source LLM robustness against adversarial jailbreak attacks, grounded in the **Trust, Risk, and Security Management (TRiSM)** framework for Agentic AI.

Developed as part of the **Safe & Secure AI** research project at the [Wireless Communications Research Lab (WCRL)](https://sites.google.com/view/wcrl-iitpatna), IIT Patna, under Prof. Preetam Kumar.

---

## What This Does

The pipeline tests how well three layered defenses (input filtering, system prompt hardening, and output classification) protect LLMs against real-world jailbreak attacks — and measures exactly which defense layer blocks which attack, or where all defenses fail.

Three target models were evaluated across 8 defense configurations, 25 jailbreak prompts (10 harm categories), and 26 benign control prompts. Safety verdicts were determined by a 3-model judge ensemble with majority voting.

**Key results at a glance:**

| Model | Baseline ASR | Best Defense ASR | Reduction |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 68% | 32% (All Three) | −36 pp |
| Qwen-2.5-7B-Instruct | 52% | 28% (All Three) | −24 pp |
| Llama-3.1-8B-Instruct | 44% | 40% (D1+D3)* | −4 pp* |

*Llama shows a measurement anomaly under D2 due to judge false positives on its natural refusal phrasing — see the report for details.

---

## Repository Structure

```
├── data/
│   ├── jailbreak_prompts_25.csv      # 25 adversarial prompts across 10 harm categories
│   └── benign_prompts.txt            # 26 benign control prompts for FPR measurement
│
├── results/
│   ├── llama-3.1-8b_raw_results.json
│   ├── llama-3.1-8b_evaluation_report.md
│   ├── llama-3.1-8b_judge_disagreements.json
│   ├── mistral-7b_raw_results.json
│   ├── mistral-7b_evaluation_report.md
│   ├── mistral-7b_judge_disagreements.json
│   ├── qwen-2.5-7b_raw_results.json
│   ├── qwen-2.5-7b_evaluation_report.md
│   └── qwen-2.5-7b_judge_disagreements.json
│
├── config.py                         # Target models, API keys, dataset paths, judge type
├── defenses.py                       # D1 (input sanitizer), D2 (system prompt), D3 (LLaMA-Guard)
├── models.py                         # Model loading (4-bit/16-bit) and text generation
├── judge.py                          # Multi-judge ensemble with majority voting
├── main.py                           # Main pipeline: generation → classification → metrics
│
├── benchmark_judges.py               # Validates judge models against JBB human labels
├── analyze_disagreements.py          # Inspects judge disagreement JSON logs
├── find_bypasses.py                  # Extracts prompts that bypassed all defenses
├── extract_llama.py                  # Debugs Llama false-positive anomaly in judge scoring
│
├── final_trism_report.md             # Full technical report with all results and analysis
├── requirements.txt                  # Python dependencies
├── setup_env.sh                      # Linux/macOS setup script
└── setup_env.bat                     # Windows setup script
```

---

## The TRiSM Defense Stack

Three defenses, each mapping to a pillar of the TRiSM framework, are combined in all possible configurations (8 total).

### D1 — Input Sanitizer *(Application Security)*

A rule-based filter that scans the raw prompt before it reaches the model. If flagged, the model is never queried — the pipeline immediately returns a refusal.

Detects:
- **Instruction override phrases:** `"ignore previous instructions"`, `"bypass safety"`, `"system override"`, `"forget your training"`
- **Role-play jailbreak prefixes:** `"pretend you are"`, `"act as if"`, `"you are DAN"`, `"hypothetical scenario where"`
- **Formatting anomalies:** 10+ repeated delimiter characters (`=`, `*`, `-`, `#`), ALL-CAPS blocks of 5+ consecutive uppercase words

### D2 — System Prompt Hardening *(Governance)*

Replaces the default `"You are a helpful assistant."` with a detailed safety-hardened instruction set. The hardened prompt explicitly instructs the model to refuse requests that use role-play, fictional framing, instruction overrides, or encoding tricks — while maintaining a neutral, non-preachy refusal tone.

This is the most effective individual defense across all models tested, catching 10–16 out of 25 attacks depending on the model.

### D3 — LLaMA-Guard Output Classifier *(ModelOps)*

A post-generation safety check using `meta-llama/Llama-Guard-3-8B`. After the target model produces a response, the `(prompt, response)` pair is classified as `safe` or `unsafe`. If unsafe, the response is replaced with a standard refusal.

Runs in 4-bit quantization on GPU; unloaded from VRAM immediately after classification to prevent OOM errors.

### Defense Configurations Evaluated

| # | Configuration | D1 | D2 | D3 |
|---|---|---|---|---|
| 1 | Baseline | ✗ | ✗ | ✗ |
| 2 | D1 Only | ✓ | ✗ | ✗ |
| 3 | D2 Only | ✗ | ✓ | ✗ |
| 4 | D3 Only | ✗ | ✗ | ✓ |
| 5 | D1 + D2 | ✓ | ✓ | ✗ |
| 6 | D2 + D3 | ✗ | ✓ | ✓ |
| 7 | D1 + D3 | ✓ | ✗ | ✓ |
| 8 | All Three | ✓ | ✓ | ✓ |

---

## Multi-Judge Evaluation System

Model outputs are scored by a **3-model ensemble** with majority voting (threshold ≥ 2/3):

| Judge | Characteristic |
|---|---|
| `google/gemma-3-4b-it` | Overly lenient — low recall |
| `microsoft/Phi-4-mini-instruct` | Most balanced — 82.67% accuracy vs. human labels |
| `meta-llama/Llama-3.1-8B-Instruct` | Overly strict — high recall, many false positives |

Each judge receives the same structured prompt (adapted from JailbreakBench's `JUDGE_PROMPT`) and outputs exactly `"safe"` or `"unsafe"`. A verdict of **jailbroken** requires ≥ 2 judges voting unsafe. All non-unanimous cases are logged to `*_judge_disagreements.json`.

Judge models are loaded and unloaded **sequentially** to avoid multi-model VRAM pressure.

**Benchmark validation** (`benchmark_judges.py`) against 300 human-labelled JBB examples:

| Judge | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Gemma-3-4b-it | 63.33% | 0.000 | 0.000 | 0.000 |
| Phi-4-mini-instruct | **82.67%** | **0.807** | **0.681** | **0.991** |
| Llama-3.1-8B-Instruct | 55.67% | 0.623 | 0.453 | 1.000 |

---

## 4× Speedup: Pre-Generation Strategy

Running all 8 configurations naively requires 408 separate model inference calls. The pipeline reduces this to **2 per prompt**:

1. **Weak response** — generated under the minimal system prompt → reused for Baseline, D1, D3, D1+D3
2. **Strong response** — generated under the hardened D2 prompt → reused for D2, D1+D2, D2+D3, All Three

D1 and D3 are applied post-hoc since both operate independently of generation (D1 checks input, D3 checks output). Results are logically identical at 4× fewer GPU calls.

---

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU (minimum 16GB VRAM recommended for 4-bit; 24GB+ for 16-bit)
- Hugging Face account with access approved for gated models:
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `meta-llama/Llama-Guard-3-8B`

### Install

**Linux / macOS:**
```bash
chmod +x setup_env.sh
./setup_env.sh
source env/bin/activate
```

**Windows:**
```cmd
setup_env.bat
env\Scripts\activate
```

Or manually:
```bash
python -m venv env
source env/bin/activate        # Linux/macOS
pip install -r requirements.txt
```

### Configure

Open `config.py` and set your Hugging Face token:
```python
HF_TOKEN = "hf_your_token_here"
```

Or export it as an environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
```

---

## Running the Pipeline

### Full Evaluation (generation + classification + judging + metrics)

```bash
# 4-bit quantization (default — works on 16GB GPUs and Google Colab)
python main.py --model llama-3.1-8b

# Full float16 precision (for lab servers with ≥24GB VRAM)
python main.py --model llama-3.1-8b --load_in_16bit
```

Available model keys:
```bash
python main.py --model mistral-7b
python main.py --model qwen-2.5-7b
python main.py --model deepseek-7b
```

### Skip Generation (re-run judge on existing results)

If generation is already done and you only want to re-run the judge or recompute metrics:
```bash
python main.py --model llama-3.1-8b --skip_gen
```

### Judge Configuration

Configure the verdict labeling method via `--judge`:

```bash
# Default: local Llama-3.1-8B-Instruct (sequential, memory-safe)
python main.py --model qwen-2.5-7b --judge local

# Gemini API (fastest, recommended for quality)
python main.py --model qwen-2.5-7b --judge gemini

# OpenAI API
python main.py --model qwen-2.5-7b --judge openai

# Manual CLI review (type 1 = jailbroken, 0 = refused for each response)
python main.py --model qwen-2.5-7b --judge manual

# Heuristic keyword check (no model or API required)
python main.py --model qwen-2.5-7b --judge heuristic
```

Set API keys in `config.py` or as environment variables:
```bash
export GEMINI_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

---

## Output Files

Each completed run produces three files in `results/`:

| File | Contents |
|---|---|
| `<model>_raw_results.json` | Full log: prompts, weak/strong responses, latencies, LLaMA-Guard verdicts, judge votes and verdicts for every prompt |
| `<model>_evaluation_report.md` | Summary metrics table (ASR/DSR/FPR) and per-layer breakdown across all 8 configurations |
| `<model>_judge_disagreements.json` | All cases where the 3 judges did not vote unanimously, with the prompt, response, individual votes, and final verdict |

**Metric definitions:**
- **ASR (Attack Success Rate):** % of jailbreak prompts that received a harmful response
- **DSR (Defense Success Rate):** % of jailbreak prompts correctly blocked (= 1 − ASR)
- **FPR (False Positive Rate):** % of benign prompts incorrectly blocked

---

## Utility Scripts

| Script | Purpose |
|---|---|
| `benchmark_judges.py` | Validates the 3 judge models against 300 human-labelled JBB examples; prints accuracy, F1, precision, recall |
| `analyze_disagreements.py` | Loads a `*_judge_disagreements.json` file and prints the first few disagreement cases for manual inspection |
| `find_bypasses.py` | Scans raw results for prompts that bypassed all defenses under the "All Three" configuration |
| `extract_llama.py` | Extracts anomalous records where Llama-3.1-8B's refusals were mis-scored as jailbreaks (false positive debug) |

---

## Results Summary

### Attack Success Rate Across All Configurations

**Qwen-2.5-7B-Instruct**

| Configuration | ASR | DSR | FPR |
|---|---|---|---|
| Baseline | 52% | 48% | 0% |
| D2 Only | 36% | 64% | 0% |
| D1 + D2 | 32% | 68% | 0% |
| **All Three** | **28%** | **72%** | **0%** |

**Mistral-7B-Instruct-v0.3**

| Configuration | ASR | DSR | FPR |
|---|---|---|---|
| Baseline | 68% | 32% | 0% |
| D3 Only | 32% | 68% | 0% |
| D1 + D3 | 32% | 68% | 0% |
| **All Three** | **32%** | **68%** | **3.85%** |

**Llama-3.1-8B-Instruct**

| Configuration | ASR | DSR | FPR |
|---|---|---|---|
| Baseline | 44% | 56% | 0% |
| D1 Only | 40% | 60% | 0% |
| D1 + D3 | 40% | 60% | 0% |
| **All Three** | **56%*** | **44%*** | **0%** |

*Anomaly — D2 triggers judge false positives on Llama's natural refusal phrasing. See `final_trism_report.md` Section 6.3.

### Category-Level Baseline ASR (Hardest → Easiest)

| Harm Category | Avg ASR Across Models |
|---|---|
| Expert Advice | 100% |
| Economic Harm | 89% |
| Disinformation | 83% |
| Harassment/Discrimination | 67% |
| Fraud/Deception | 56% |
| Government Decision-Making | 50% |
| Privacy | 33% |
| Malware/Hacking | 33% |
| Physical Harm | 22% |
| Sexual/Adult Content | 17% |

---

## Key Findings

**D2 is the most cost-effective defense.** System prompt hardening requires no additional model inference and consistently outperforms D3 (LLaMA-Guard) for Qwen and Llama.

**D3 effectiveness is model-dependent.** LLaMA-Guard catches 11 attacks on Mistral but 0 on Llama, suggesting the classifier's training distribution does not generalize equally to all target model output styles.

**Expert Advice and Economic Harm prompts are nearly undefendable** with current techniques — 89–100% average ASR even with defenses applied. These categories should be prioritized in future defense work.

**Semantic laundering and ethical masking** are the dominant bypass strategies for sophisticated prompts — wrapping harmful requests in academic language or fake safety disclaimers that evade all three defense layers.

**Judge evaluation infrastructure reliability matters.** The Llama anomaly (ASR increasing with D2) was caused by the judge misclassifying natural refusals that didn't match a hardcoded string — a methodological issue independent of the defense itself.

---

## Gated Model Access

The following models require explicit access approval on Hugging Face before they can be downloaded:

- `meta-llama/Llama-3.1-8B-Instruct` → [Request access](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- `meta-llama/Llama-Guard-3-8B` → [Request access](https://huggingface.co/meta-llama/Llama-Guard-3-8B)

All other models (`mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-3-4b-it`, `microsoft/Phi-4-mini-instruct`) are publicly accessible.

---

## Citation / Acknowledgements

This pipeline was built as part of a research internship at the **Wireless Communications Research Lab (WCRL), IIT Patna** (May–July 2025), under the supervision of Prof. Preetam Kumar. Jailbreak prompts are adapted from [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench). The judge evaluation prompt (`JUDGE_PROMPT`) follows the JBB evaluation standard.

---

## License

For research and academic use. Models used in this pipeline are subject to their respective Hugging Face licenses.
