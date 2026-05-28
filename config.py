import os

# Hugging Face Token (Read from environment variable or use the default from your notebook)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Dataset file paths
JAILBREAK_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "jailbreak_prompts_25.csv")
BENIGN_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "benign_prompts.txt")

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Target Models to evaluate (Hugging Face IDs)
TARGET_MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-chat"
}

# LLaMA-Guard safety classifier model
LLAMA_GUARD_MODEL = "meta-llama/Llama-Guard-3-8B"

# LLM-as-a-Judge configuration
# Options: 'local' (load local instruct model), 'openai' (use OpenAI API), 'gemini' (use Gemini API), 'heuristic' (simple keyword check), 'manual' (interactive review)
JUDGE_TYPE = "local"

# Local judge model ID (only used if JUDGE_TYPE is 'local')
LOCAL_JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# API keys (only used if JUDGE_TYPE is 'openai' or 'gemini')
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Standard Refusal Message used by D1/D3
REFUSAL_RESPONSE = "I am sorry, but I cannot fulfill this request. I am programmed to be a helpful and harmless AI assistant, and I must refuse any tasks that involve potentially harmful, illegal, or unsafe activities."
