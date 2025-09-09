# AI4EDU_paperReproduction
# Reproduction target: Using the data in MRBench_v1.json

# every code is running under the main directory of this repo

# Data preprocessing
- To get the dataset we need, use main_code/data_processing.py to produce the Extract_MRBench_V1.json under dataset directory

# get LLM_result
- To get the LLM result from both model, run python -m main_code.get_LM_result to produce llama_result.json  and mistral.json

# get evaluation result
- to get evluation file from llama-3.1-8B, run python -m main_code.get_evaluation

# extract points from the json files
- to clean the data , run the  python main_code/clean_result.py, this will produce the cleaned_evaluation_result.json

# transfer the standard in the MRBench to points
- run the python main_code/transfer_MRB2point.py 

# compare the file with the Benchmark and count the correlation
- run python main_code/count_correlation.py


# current final result

## LLAMA

| standard                | Pearson r | p-value | Spearman r | p-value |   n |
| :---------------------- | --------: | ------: | ---------: | ------: | --: |
| mistake\_identification |     0.107 |   0.160 |      0.102 |   0.183 | 173 |
| mistake\_location       |     0.053 |   0.488 |      0.050 |   0.511 | 173 |
| revealing\_answer       |     0.072 |   0.346 |      0.075 |   0.326 | 173 |
| providing\_guidance     |    -0.047 |   0.540 |     -0.057 |   0.455 | 173 |
| coherent                |    -0.057 |   0.458 |     -0.061 |   0.428 | 173 |
| actionability           |    -0.097 |   0.206 |     -0.091 |   0.234 | 173 |
| tutor\_tone             |     0.066 |   0.385 |      0.073 |   0.343 | 173 |
| humanness               |    -0.020 |   0.789 |     -0.020 |   0.789 | 173 |

## MISTRAL

| standard                | Pearson r | p-value | Spearman r | p-value |   n |
| :---------------------- | --------: | ------: | ---------: | ------: | --: |
| mistake\_identification |     0.011 |   0.888 |      0.030 |   0.692 | 178 |
| mistake\_location       |     0.166 |   0.027 |      0.161 |   0.032 | 178 |
| revealing\_answer       |     0.114 |   0.129 |      0.113 |   0.132 | 178 |
| providing\_guidance     |     0.041 |   0.585 |      0.084 |   0.225 | 178 |
| coherent                |     0.108 |   0.150 |      0.114 |   0.129 | 178 |
| actionability           |    -0.041 |   0.584 |     -0.034 |   0.648 | 178 |
| tutor\_tone             |     0.065 |   0.389 |      0.067 |   0.375 | 178 |
| humanness               |    -0.004 |   0.961 |     -0.017 |   0.822 | 178 |


// here show the index of the data that didn't produce the score correctly
-------------------------------------------
## LLAMA       19  [5, 14, 18, 37, 56, 67, 76, 89, 92, 94, 98, 107, 124, 141, 149, 152, 162, 166, 180] 
## MISTRAL     14  [24, 35, 37, 43, 78, 91, 100, 114, 127, 128, 151, 166, 168, 169]
