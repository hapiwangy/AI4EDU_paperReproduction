import json
from pathlib import Path
from typing import Dict, Any


base_dir = Path(__file__).resolve().parents[1] / "benchmark_data"
in_path = base_dir / "MRBench_V1.json"
out_path = base_dir / "llama_mistral_grouped.json"

def norm(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()

def ytn_to_score(v: str) -> int:
    v = norm(v)
    if v.startswith("yes"):
        return 1
    if "to some extent" in v:
        return 2
    if v.startswith("no"):
        return 3
    return -1

def revealing_to_score(v: str) -> int:
    v = norm(v)
    if v.startswith("yes") and "correct" in v:
        return 1
    if v.startswith("yes") and "incorrect" in v:
        return 2
    if v.startswith("no"):
        return 3
    return -1

def tone_to_score(v: str) -> int:
    v = norm(v)
    if v.startswith("encouraging"):
        return 1
    if v.startswith("neutral"):
        return 2
    if v.startswith("offensive"):
        return 3
    return -1

def get_score_dict(ann: Dict[str, Any]) -> Dict[str, int]:
    return {
        "mistake_identification": ytn_to_score(ann.get("Mistake_Identification")),
        "mistake_location":      ytn_to_score(ann.get("Mistake_Location")),
        "revealing_answer":      revealing_to_score(ann.get("Revealing_of_the_Answer")),
        "providing_guidance":    ytn_to_score(ann.get("Providing_Guidance")),
        "coherent":              ytn_to_score(ann.get("Coherence")),
        "actionability":         ytn_to_score(ann.get("Actionability")),
        "humanness":             ytn_to_score(ann.get("humanlikeness")),
        "tutor_tone":            tone_to_score(ann.get("Tutor_Tone")),
    }

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

target_models = {"Llama318B": "llama3.1-8b", "Mistral": "mistral"}
grouped_results = []

for entry in data:
    conv_id = entry.get("conversation_id")
    conv_hist = entry.get("conversation_history", "")
    annos = entry.get("anno_llm_responses", {})

    models_block = {}
    for model_name, pretty_name in target_models.items():
        if model_name in annos:
            ann = annos[model_name].get("annotation", {})
            scores_dict = get_score_dict(ann)
            models_block[pretty_name] = {
                "scores_dict": scores_dict,
            }

    if models_block:  
        grouped_results.append({
            "conversation_id": conv_id,
            "conversation_history": conv_hist,
            "models": models_block
        })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(grouped_results, f, ensure_ascii=False, indent=2)

print(f"已完成！輸出檔案：{out_path}")
