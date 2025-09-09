import os
import json
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr


CURRENT_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "dataset"))
BENCH_DIR   = os.path.abspath(os.path.join(CURRENT_DIR, "..", "benchmark_data"))

CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_evaluation_result.json")
ORG_PATH     = os.path.join(BENCH_DIR,   "llama_mistral_grouped.json")


with open(CLEANED_PATH, "r", encoding="utf-8") as f:
    cleaned = json.load(f)
with open(ORG_PATH, "r", encoding="utf-8") as f:
    org = json.load(f)

MAPPING = {
    "mistake_identification_point": "mistake_identification",
    "mistake_location_point": "mistake_location",
    "revealing_answer_point": "revealing_answer",
    "providing_guidance_point": "providing_guidance",
    "coherent_point": "coherent",
    "actionability_point": "actionability",
    "tutor_tone_point": "tutor_tone",
    "humanness_point": "humanness",
}
STANDARDS = list(MAPPING.values())

def normalize_model(name: str) -> str:
    n = (name or "").lower()
    if "llama" in n:
        return "llama"
    if "mistral" in n:
        return "mistral"
    return n

def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if method == "pearson":
                return pearsonr(x, y)
            else:
                return spearmanr(x, y)
        except Exception:
            return np.nan, np.nan


def fmt_num(v):
    return "NaN" if pd.isna(v) else f"{v:.3f}"

def print_corr_table(model_name, rows):
    """
    rows: list of dicts with keys:
      standard, pearson_r, pearson_p, spearman_r, spearman_p, n_pairs
    """
    std_w = max(len("standard"), max(len(r["standard"]) for r in rows)) + 2
    num_w = 10  
    n_w   = 5

    header = (
        f"{'standard'.ljust(std_w)}"
        f"{'Pearson r'.rjust(num_w)}  {'p-value'.rjust(num_w)}  "
        f"{'Spearman r'.rjust(num_w)}  {'p-value'.rjust(num_w)}  "
        f"{'n'.rjust(n_w)}"
    )
    line = "-" * len(header)

    print(f"\n=== {model_name.upper()} ===")
    print(header)
    print(line)
    for r in rows:
        print(
            f"{r['standard'].ljust(std_w)}"
            f"{fmt_num(r['pearson_r']).rjust(num_w)}  {fmt_num(r['pearson_p']).rjust(num_w)}  "
            f"{fmt_num(r['spearman_r']).rjust(num_w)}  {fmt_num(r['spearman_p']).rjust(num_w)}  "
            f"{str(int(r['n_pairs'])).rjust(n_w)}"
        )

def print_skipped_table(skipped_indices_by_model):
    model_w = max(len("Model"), *(len(k.upper()) for k in skipped_indices_by_model.keys() or ["LLAMA","MISTRAL"]))
    count_w = max(len("Count"), 5)
    header = f"{'Model'.ljust(model_w)}  {'Count'.rjust(count_w)}  Indices"
    line = "-" * (len(header) + 20)

    print("\n=== Skipped element indices (by model) ===")
    print(header)
    print(line)
    for m in ("llama", "mistral"):
        idxs = sorted(skipped_indices_by_model.get(m, []))
        idx_str = "[" + ", ".join(map(str, idxs)) + "]"
        print(f"{m.upper().ljust(model_w)}  {str(len(idxs)).rjust(count_w)}  {idx_str}")


org_lookup = {item.get("conversation_history"): item for item in org}

rows = []
skipped_indices_by_model = defaultdict(set)  # { "llama": {idx1, ...}, "mistral": {...} }

for idx, c in enumerate(cleaned):
    conv = c.get("conversation_history")
    if not conv or conv not in org_lookup:
        continue

    o = org_lookup[conv]
    o_models = (o.get("models") or {})
    o_norm_map = defaultdict(list)
    for mk in o_models.keys():
        o_norm_map[normalize_model(mk)].append(mk)

    for c_model, c_payload in (c.get("anno_llm_responses") or {}).items():
        c_norm = normalize_model(c_model)
        if c_norm not in o_norm_map:
            continue

        o_model_key = o_norm_map[c_norm][0]
        c_scores = (c_payload or {}).get("annotation") or {}
        o_scores = (o_models[o_model_key] or {}).get("scores_dict") or {}

        if any(c_scores.get(k, None) == -1 for k in MAPPING.keys()):
            skipped_indices_by_model[c_norm].add(idx)
            continue

        row = {"conversation_history": conv, "model": c_norm}
        for c_key, o_key in MAPPING.items():
            row[f"cleaned_{o_key}"] = c_scores.get(c_key, np.nan)
            row[f"org_{o_key}"]     = o_scores.get(o_key, np.nan)
        rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    print("No aligned rows to compute (all valid rows skipped or unmatched).")
else:
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].copy()
        out_rows = []
        for std in STANDARDS:
            a = f"cleaned_{std}"
            b = f"org_{std}"
            pairs = sub[[a, b]].dropna()
            pr, pp = safe_corr(pairs[a].values, pairs[b].values, method="pearson")
            sr, sp = safe_corr(pairs[a].values, pairs[b].values, method="spearman")
            out_rows.append({
                "standard": std,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "n_pairs": len(pairs)
            })
        print_corr_table(model, out_rows)


print_skipped_table(skipped_indices_by_model)
