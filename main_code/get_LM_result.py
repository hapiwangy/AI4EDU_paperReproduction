import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from helper import prompts

model_ids = {"meta-llama/Llama-3.1-8B-Instruct":"llama", "mistralai/Mistral-7B-Instruct-v0.1":"mistral"}
dataset = "Extract_MRBench_V1.json"
output_file = "result.json"


current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)
dataset_dir = os.path.join(current_dir, "..", "dataset")
dataset_dir = os.path.abspath(dataset_dir)   


MathDialPrompt = ""
BridgePrompt = ""
with open(os.path.join(dataset_dir, "prompt_Bridge.txt"), "r", encoding="utf-8") as f:
    BridgePrompt = f.read()

with open(os.path.join(dataset_dir, "prompt_MathDial.txt"), "r", encoding="utf-8") as f:
    MathDialPrompt = f.read()

current_json_file = os.path.join(dataset_dir, dataset)
with open(current_json_file, "r", encoding="utf-8") as fp:
    json_data = json.load(fp)

def _take_text(g):
    if isinstance(g, list):
        return g[0]['generated_text']
    return g['generated_text']
for model_id, model_name in model_ids.items():
    final_result = []
    org = True
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto"
    ).eval()
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    idx_list, prompt_list = [], []
    for x in range(len(json_data)):
        cur_data = json_data[x]
        prompt = prompts.MathDial_Prompt(MathDialPrompt, cur_data, org) if cur_data['Data'] == "MathDial" else prompts.Bridge_Prompt(BridgePrompt, cur_data, org)
        idx_list.append(x)
        prompt_list.append(prompt)
    batch_size = 32  # change this number due to the GPU
    for start in tqdm(range(0, len(prompt_list), batch_size), desc="Generating", unit="batch"):
        end = min(start + batch_size, len(prompt_list))
        batch_prompts = prompt_list[start:end]
        batch_idxs = idx_list[start:end]

        generations = generator(
            batch_prompts,
            batch_size=len(batch_prompts),   
            generation_config=gen_cfg,
            return_full_text=False,
            truncation=True,
        )

        for x, g in zip(batch_idxs, generations):
            cur_data = json_data[x]
            temp = {}
            result = prompts.safe_cut_at_first_heading(_take_text(g))
            temp["result"] = result
            temp["Data"] = cur_data["Data"]
            temp["conversation_history"] = cur_data["conversation_history"]
            temp["Topic"] = cur_data["Topic"]
            temp["Ground_Truth_Solution"] = cur_data["Ground_Truth_Solution"]
            final_result.append(temp)
            print(len(final_result))
    with open(os.path.join(dataset_dir, f"{model_name}_{output_file}"), "w+", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
