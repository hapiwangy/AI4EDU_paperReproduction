import os
import json
from pathlib import Path
from collections import OrderedDict


current_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(current_dir, "..", "dataset")
dataset_dir = os.path.abspath(dataset_dir)

src_path = Path(os.path.join(dataset_dir, "evaluation_result.json"))
out_path = Path(os.path.join(dataset_dir, "cleaned_evaluation_result.json"))

with src_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

grouped = OrderedDict()

for row in data:
    source_index = row.get("source_index", None)
    source_file = row.get("source_file", "").strip()
    model_name = source_file.split("_")[0] if "_" in source_file else Path(source_file).stem

    if source_index not in grouped:
        grouped[source_index] = {
            "source_index": source_index,
            "conversation_history": row.get("conversation_history", "").strip(),
            "anno_llm_responses": OrderedDict()
        }

    grouped[source_index]["anno_llm_responses"][model_name] = {
        "response": row.get("response", None),
        "annotation": row.get("annotation", {}),
        "source_file": source_file
    }


result = list(grouped.values())

with out_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(result)} items to {out_path}")