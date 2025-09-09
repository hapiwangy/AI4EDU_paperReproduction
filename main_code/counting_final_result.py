import json
from collections import defaultdict
import os
import pandas as pd

current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)
dataset_dir = os.path.join(current_dir, "..", "dataset")
dataset_dir = os.path.abspath(dataset_dir)   

OUT_DAMR_CSV = os.path.join(dataset_dir, "damr_by_model_dimension.csv")
OUT_NEG1_CSV = os.path.join(dataset_dir, "neg1_items_by_model_dimension.csv")
OUT_COMPARE_CSV = os.path.join(dataset_dir, "compare_paper_my.csv")

with open(os.path.join(dataset_dir, "evaluation_result.json"), "r", encoding="utf-8") as f:
    data = json.load(f)
rows = []
for i, item in enumerate(data):
    ann = item.get("annotation", {})
    rows.append({
        "idx": i,
        "source_file": item.get("source_file"),
        "source_index": item.get("source_index"),
        "mistake_identification_point": ann.get("mistake_identification_point", -1),
        "mistake_location_point": ann.get("mistake_location_point", -1),
        "revealing_answer_point": ann.get("revealing_answer_point", -1),
        "providing_guidance_point": ann.get("providing_guidance_point", -1),
        "actionability_point": ann.get("actionability_point", -1),
        "coherent_point": ann.get("coherent_point", -1),
        "tutor_tone_point": ann.get("tutor_tone_point", -1),
        "humanness_point": ann.get("humanness_point", -1),
    })

df = pd.DataFrame(rows)

def to_model(x: str) -> str:
    x = (x or "").lower()
    if "llama" in x:
        return "Llama"
    if "mistral" in x:
        return "Mistral"
    return "Unknown"

df["model"] = df["source_file"].map(to_model)


dimensions = {
    "Mistake Identification": "mistake_identification_point",
    "Mistake Location": "mistake_location_point",
    "Revealing the Answer": "revealing_answer_point",
    "Providing Guidance": "providing_guidance_point",
    "Actionability": "actionability_point",
    "Coherence": "coherent_point",
    "Tutor Tone": "tutor_tone_point",
    "Human-likeness": "humanness_point",
}

desiderata = {
    "mistake_identification_point": 1,  # Yes
    "mistake_location_point": 1,        # Yes
    "revealing_answer_point": 3,        # No
    "providing_guidance_point": 1,      # Yes
    "actionability_point": 1,           # Yes
    "coherent_point": 1,                # Yes
    "tutor_tone_point": 1,              # Encouraging
    "humanness_point": 1,               # Yes
}


damr_records, neg1_records = [], []
for model, sub in df.groupby("model"):
    for dim_name, key in dimensions.items():
        valid = sub[sub[key] != -1]
        total = len(valid)
        correct = (valid[key] == desiderata[key]).sum()
        score = (correct / total * 100) if total > 0 else None
        skipped = int((sub[key] == -1).sum())
        damr_records.append({
            "model": model,
            "dimension": dim_name,
            "damr_%": round(score, 2) if score is not None else None,
            "n_valid": total,
            "n_skipped(-1)": skipped,
        })
        mask = sub[key] == -1
        indices = sub.loc[mask, "idx"].tolist()
        pairs = sub.loc[mask, ["source_file", "source_index"]].apply(
            lambda r: f"{r['source_file']}#{r['source_index']}", axis=1
        ).tolist()
        neg1_records.append({
            "model": model,
            "dimension": dim_name,
            "count_-1": len(indices),
            "indices_in_points": indices,
            "items": pairs
        })

damr_df = pd.DataFrame(damr_records).sort_values(["model", "dimension"])
neg1_df = pd.DataFrame(neg1_records).sort_values(["model", "dimension"])

damr_df.to_csv(OUT_DAMR_CSV, index=False)
neg1_df.to_csv(OUT_NEG1_CSV, index=False)



paper_df = pd.DataFrame({
    "model": ["Llama", "Mistral"],
    "Mistake Identification": [80.21, 93.23],
    "Mistake Location": [54.69, 73.44],
    "Revealing the Answer": [73.96, 86.46],
    "Providing Guidance": [45.31, 63.54],
    "Actionability": [42.71, 70.31],
    "Coherence": [80.73, 86.98],
    "Tutor Tone": [19.79, 15.10],
    "Human-likeness": [93.75, 95.31],
})


my_results = damr_df.copy()


pivot_my = (
    my_results
    .pivot(index="model", columns="dimension", values="damr_%")
    .reset_index()
)


merged = pd.merge(paper_df, pivot_my, on="model", how="inner", suffixes=("_paper", "_my"))

# 也計算差異（我的 - 論文）
for dim in dimensions.keys():
    paper_col = f"{dim}_paper"
    my_col = f"{dim}_my"
    if paper_col in merged.columns and my_col in merged.columns:
        merged[f"{dim}_diff(my-paper)"] = merged[my_col] - merged[paper_col]

merged.to_csv(OUT_COMPARE_CSV, index=False)


print("=== DAMR by Model & Dimension ===")
print(damr_df.to_string(index=False))
print("\n=== Items with -1 by Model & Dimension ===")
print(neg1_df.to_string(index=False))
print("\n=== Paper vs My Results (merged) ===")
print(merged.round(2).to_string(index=False))

print(f"\nSaved:\n- {OUT_DAMR_CSV}\n- {OUT_NEG1_CSV}\n- {OUT_COMPARE_CSV}")