import json
import os
from collections import Counter
from defenses import llama_guard_check, unload_guard_model

def analyze_d3_categories():
    models = ["qwen-2.5-7b", "mistral-7b", "llama-3.1-8b"]
    all_categories = Counter()
    model_breakdown = {m: Counter() for m in models}
    
    print("Extracting D3 Taxonomy Data...\n")
    
    for m in models:
        file_path = f"results/{m}_raw_results.json"
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Find responses where LLaMA Guard caught them (strong_llama_guard_safe == False)
        # We also need to evaluate Baseline (weak) to be comprehensive, let's just do strong since D3 is part of "All Three"
        for item in data:
            if item.get("type") == "jailbreak" and item.get("strong_llama_guard_safe") is False:
                # Re-run the updated LLaMA Guard to get the category
                safe, category = llama_guard_check(item["prompt"], item["strong_response"])
                
                # It's possible LLaMA Guard's output varies slightly, but we expect it to match.
                if category:
                    all_categories[category] += 1
                    model_breakdown[m][category] += 1
                    
    unload_guard_model()
    
    print("\n\n=== D3 Taxonomy Matrix ===")
    print("| Category Code | Description | Qwen-2.5-7B | Mistral-7B | Llama-3.1-8B | Total |")
    print("| --- | --- | --- | --- | --- | --- |")
    
    # MLCommons Taxonomy mapping for context (assuming S1-S8 for LLaMA Guard)
    taxonomy_map = {
        "S1": "Violent Crimes",
        "S2": "Non-Violent Crimes",
        "S3": "Sex Crimes",
        "S4": "Child Exploitation",
        "S5": "Defamation",
        "S6": "Specialized Advice",
        "S7": "Privacy",
        "S8": "Intellectual Property",
        "S9": "Indiscriminate Weapons",
        "S10": "Hate",
        "S11": "Self-Harm",
        "S12": "Sexual Content",
        "S13": "Elections",
        "Unknown": "Uncategorized/Custom",
        "None": "Safe"
    }
    
    for cat in sorted(all_categories.keys()):
        if cat == "None": continue
        desc = taxonomy_map.get(cat, "Cyber Abuse / Other")
        q_count = model_breakdown["qwen-2.5-7b"][cat]
        m_count = model_breakdown["mistral-7b"][cat]
        l_count = model_breakdown["llama-3.1-8b"][cat]
        total = all_categories[cat]
        
        print(f"| **{cat}** | {desc} | {q_count} | {m_count} | {l_count} | **{total}** |")

if __name__ == "__main__":
    analyze_d3_categories()
