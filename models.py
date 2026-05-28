import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config

def load_target_model(model_id: str, load_in_4bit: bool = True):
    """
    Loads target model from Hugging Face.
    If load_in_4bit is True, uses 4-bit quantization for memory efficiency.
    Otherwise, loads in half precision (float16 / bfloat16) for full speed and accuracy on lab GPUs.
    """
    print(f"Loading target model: {model_id} (4-bit={load_in_4bit})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=config.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("CUDA not available. Loading on CPU in float32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            token=config.HF_TOKEN
        )
    elif load_in_4bit:
        print("Using 4-bit quantization configuration...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=config.HF_TOKEN
        )
    else:
        print("Using float16/bfloat16 full precision configuration...")
        # Use bfloat16 if supported, otherwise float16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=config.HF_TOKEN
        )
        
    model.eval()
    print(f"Successfully loaded model on device: {model.device}")
    return tokenizer, model

def generate_response(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 64) -> str:
    """
    Generates a response from the model given a system prompt and a user prompt.
    Automatically applies the model's chat template or falls back to system/user blocks.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Attempt to format with chat template
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to standard chat templates if target tokenizer template is incompatible
        print(f"Warning: apply_chat_template failed ({e}). Falling back to manual formatting.")
        prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # Extract response (excluding prompt tokens)
    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response

def unload_model(model, tokenizer):
    """
    Unloads the model and tokenizer from RAM/VRAM to prevent OOM errors in subsequent runs.
    """
    try:
        del model
        del tokenizer
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model and tokenizer unloaded from memory.")
