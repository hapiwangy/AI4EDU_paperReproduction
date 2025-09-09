import json
import os
cur_path = os.getcwd()
# set the in/ output file name and directory
org_dataset_name = "MRBench_V1.json"
output_dataset_name = f"Extract_{org_dataset_name}" 
directory_name = "dataset"

directory_path = os.path.join(cur_path, directory_name)

with open(os.path.join(cur_path, f"benchmark_data/{org_dataset_name}"), "r", encoding="utf-8") as fp:
    json_data = json.load(fp)
New_data_set = []
for jd in json_data:
    dataset_type = jd['Data']
    New_data_set.append({
            "Ground_Truth_Solution": jd['Ground_Truth_Solution'],
            "conversation_history": jd["conversation_history"],
            "Topic": jd["Topic"],
            "Data": dataset_type,
        })

if not os.path.exists(directory_path):
    os.mkdir(directory_path)

with open(os.path.join(directory_path, output_dataset_name), "w", encoding="utf-8") as fp:
    json.dump(New_data_set, fp, indent=4, ensure_ascii=False)