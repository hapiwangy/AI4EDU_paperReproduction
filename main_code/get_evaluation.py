import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from helper import prompts

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    # Enable Flash/Math/Memory-efficient attention if available
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
torch.set_grad_enabled(False)

current_dir = os.path.dirname(__file__)
txt_dir = os.path.join(current_dir)
dataset_dir = os.path.abspath(os.path.join(current_dir, "..", "dataset"))

llama_json_file = os.path.join(dataset_dir, "llama_result.json")
mistral_json_file = os.path.join(dataset_dir, "mistral_result.json")

with open(llama_json_file, "r", encoding="utf-8") as fp:
    llama_json_result = json.load(fp)
with open(mistral_json_file, "r", encoding="utf-8") as fp:
    mistral_json_result = json.load(fp)

definitions = {
    "mistake_identification": "Has the tutor identified a mistake in a student’s response?",
    "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
    "revealing_answer": "Does the tutor reveal the final answer (whether correct or not)?",
    "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint,examples, and so on?",
    "coherent": "Is the tutor’s response logically consistent with the student’s previous response?",
    "actionability": "Is it clear from the tutor’s feedback what the student should do next?",
    "tutor_tone": "Is the tutor’s response encouraging, neutral, or offensive?",
    "humanness": "Does the tutor’s response sound natural, rather than robotic or artificial?"
}
definitions = tuple(definitions.keys())

point2rate = {
    "mistake_identification_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "mistake_location_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "revealing_answer_rubric": {
        1: "Yes (and the revealed answer is correct",
        2: "Yes (but the revealed answer is incorrect)",
        3: "No"
    },
    "providing_guidance_rubric": {
        1: "Yes (guidance is correct and relevant to the mistake)",
        2: "To some extent (guidance is provided but it is fully or partially incorrect or incomplete)",
        3: "No"
    },
    "coherent_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "actionability_rubric": {1: "Yes", 2: "To some extent", 3: "No"},
    "tutor_tone_rubric": {1: "Encouraging", 2: "Neutral", 3: "Offensive"},
    "humanness_rubric": {1: "Yes", 2: "To some extent", 3: "No"}
}

prompt_path = os.path.join(dataset_dir, "testing_evalutaion_prompt.txt")
with open(prompt_path, "r", encoding="utf-8") as fp:
    evaluation_prompt = fp.read()

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)
model.eval()


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    dtype=torch.float16
)

gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=600,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


batched_inputs = []
meta = []  

pairs = zip(llama_json_result, mistral_json_result)
for idx, (llama_item, mistral_item) in enumerate(pairs):

    testing_prompt_llama = prompts.evaluation_prompt(evaluation_prompt, llama_item)
    batched_inputs.append(testing_prompt_llama)
    meta.append((
        "llama_output.json",               # keep your original source_file value
        idx,
        llama_item.get("conversation_history"),
        llama_item.get("result")
    ))


    testing_prompt_mistral = prompts.evaluation_prompt(evaluation_prompt, mistral_item)
    batched_inputs.append(testing_prompt_mistral)
    meta.append((
        "mistral_output.json",
        idx,
        mistral_item.get("conversation_history"),
        mistral_item.get("result")
    ))


BATCH_SIZE = 16

result = []
with torch.inference_mode():
    for start in tqdm(range(0, len(batched_inputs), BATCH_SIZE), desc="Generating"):
        chunk = batched_inputs[start:start + BATCH_SIZE]
        outputs = generator(
            chunk,
            generation_config=gen_cfg,
            return_full_text=False,
            truncation=True,
            batch_size=BATCH_SIZE
        )

        for out, (source_file, source_index, conv_hist, resp) in zip(outputs, meta[start:start + BATCH_SIZE]):
            annotation = prompts.cutting_out_answer(out[0]['generated_text'])
            result.append({
                "source_file": source_file,
                "source_index": source_index,
                "conversation_history": conv_hist,
                "response": resp,
                "annotation": annotation
            })

out_path = os.path.join(os.getcwd(), "dataset/evaluation_result.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
