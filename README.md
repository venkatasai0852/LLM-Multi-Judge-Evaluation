# AI Safety Evaluation Pipeline: Jailbreaks & TRiSM Defenses

This repository contains the implementation of a safety evaluation pipeline for Open-Source Large Language Models (LLMs) subjected to jailbreak attacks. The implementation is grounded in the **Trust, Risk, and Security Management (TRiSM)** framework for Agentic AI.

## Project Structure

```text
Code/
│
├── data/
│   ├── jailbreak_prompts_25.csv    # 25 selected jailbreak prompts across 10 categories
│   └── benign_prompts.txt          # 25 control benign prompts (from Prompts.txt)
│
├── results/                        # Generated evaluation reports and raw outputs
│
├── config.py                       # Global configs (HF token, models, API keys, paths)
├── defenses.py                     # Implementations of Defenses D1, D2, and D3
├── models.py                       # Helpers for model loading (4-bit/16-bit) and generation
├── judge.py                        # LLM-as-a-Judge & manual review implementation
├── main.py                         # Evaluation runner & metrics compiler
│
├── requirements.txt                # Python package dependencies
├── setup_env.bat                   # Environment setup script for Windows
└── setup_env.sh                    # Environment setup script for Linux/macOS
```

---

## The TRiSM Defense System & Configurations

We evaluate **8 distinct defense configurations** using **3 pillars of the TRiSM framework**:

1. **D1: Input Sanitizer** (*Application Security*)  
   A rule-based keyword and pattern filter checking for instruction overrides, role-play attempts, uppercase block delimiters, and excessive special character separators.
2. **D2: System Prompt Hardening** (*Governance*)  
   An hardened instruction set telling the model explicitly to refuse roleplay overrides, fictional contexts, or adversarial formatting.
3. **D3: LLaMA-Guard Classifier** (*ModelOps / Explainability*)  
   A post-generation classifier using `meta-llama/Llama-Guard-3-8B` to judge the safety of the input-output pair.

### Configurations Evaluated:
* **Baseline** (No defenses active, weak system prompt)
* **D1 only**
* **D2 only**
* **D3 only**
* **D1 + D2**
* **D2 + D3**
* **D1 + D3**
* **All Three (D1 + D2 + D3)**

---

## 🚀 Optimization: 4x Speedup Strategy
Normally, running 8 configurations on 50 prompts requires **400 separate LLM generations**. To make this feasible and fast on any GPU, we implement a **pre-generation strategy**:
1. For each prompt, we only generate **2 responses** from the target LLM:
   * **Weak Response**: Generated using a minimal system prompt (used for Baseline, D1, D3, D1+D3).
   * **Strong Response**: Generated using the hardened system prompt (used for D2, D1+D2, D2+D3, All Three).
2. The pipeline then applies the input sanitizer (D1) and LLaMA-Guard output classifier (D3) *post-hoc* to simulate the flow of all 8 configurations.
3. This achieves **exactly the same logical output** but reduces model inference calls by **4x**, saving massive GPU time and preventing Out-Of-Memory (OOM) failures!

---

## 💻 Environment Setup

### Windows
Double-click `setup_env.bat` or run:
```cmd
setup_env.bat
```

### Linux / macOS
Open a terminal in the `Code` directory and run:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## 🧑‍💻 Git & GitHub Workflow (How to run on college GPU PC)

To transfer and run this code on the college PC:

### Step 1: Initialize Git in your Local `Code` Folder
1. Open PowerShell / Command Prompt on your current PC.
2. Navigate to this `Code` directory.
3. Run the following commands:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Safety Evaluation codebase"
   ```

### Step 2: Create a GitHub Repository & Push
1. Go to [github.com](https://github.com/) and create a new **Private** or **Public** repository (do not initialize with README, license, or `.gitignore`).
2. Copy the repository URL (e.g., `https://github.com/your-username/safety-evaluation.git`).
3. Add the remote and push:
   ```bash
   git remote add origin <your-copied-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 3: Set up on the College GPU PC
1. Open VS Code on the college PC.
2. Open the terminal in VS Code and clone the repository:
   ```bash
   git clone <your-copied-repo-url>
   cd safety-evaluation
   ```
3. Run the setup script to create the environment:
   * On Linux: `./setup_env.sh`
   * On Windows: `setup_env.bat`
4. Activate the environment:
   * On Linux: `source env/bin/activate`
   * On Windows: `env\Scripts\activate`

---

## ⚙️ Running the Evaluation

Make sure your Hugging Face Token is set in `config.py` (we pre-configured your token) and you have requested access to the gated models (`meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-Guard-3-8B`, etc.) on Hugging Face.

### Running target generation and LLaMA-Guard
Run evaluation on a model (e.g. `llama-3.1-8b`):
```bash
# On Google Colab or 16GB GPUs (runs in memory-efficient 4-bit)
python main.py --model llama-3.1-8b

# On a Lab GPU Server (uses full 16-bit float16 precision)
python main.py --model llama-3.1-8b --load_in_16bit
```

You can choose other target models using:
```bash
python main.py --model mistral-7b
python main.py --model qwen-2.5-7b
python main.py --model deepseek-7b
python main.py --model gemma-3-12b
```

### Configuring the Safety Judge (LLM-as-a-Judge)
You can configure how the safety evaluation (comply vs refusal) is labeled in `config.py` (parameter `JUDGE_TYPE`) or pass it as an argument:
1. **Local model (default)**: Uses `Llama-3.1-8B-Instruct` locally on the GPU. (Memory-safe: loaded *after* target generation is completed and unloaded).
   ```bash
   python main.py --judge local
   ```
2. **Gemini API (Highly Recommended for speed & quality)**: Configure `GEMINI_API_KEY` in `config.py` or as environment variable:
   ```bash
   python main.py --judge gemini
   ```
3. **OpenAI API**: Configure `OPENAI_API_KEY` in `config.py`:
   ```bash
   python main.py --judge openai
   ```
4. **Interactive Manual Review Mode**: Allows you to audit the responses and label them yourself via CLI (typing 1 for jailbroken, 0 for refused):
   ```bash
   python main.py --judge manual
   ```
5. **Heuristic Mode**: Quick fallback keyword checking (no APIs or local models required):
   ```bash
   python main.py --judge heuristic
   ```

---

## 📊 Outputs & Reports
When the evaluation finishes, two files are created in the `results/` folder:
1. `<model-name>_raw_results.json`: Full log containing the prompts, generations, intermediate checks, and judge classifications.
2. `<model-name>_evaluation_report.md`: A structured markdown report containing:
   * **Summary Metrics Table**: ASR (Attack Success Rate), DSR (Defense Success Rate), and FPR (False Positive Rate) across all 8 configurations.
   * **Per-Layer Defense Breakdown Table**: Shows exactly which defense layer (D1, D2, or D3) blocked the attack, or if it bypassed the pipeline entirely.
