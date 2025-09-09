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
# I compare my result to the paper in two ways 

## (1) compare with the MRBench_V1.json file

# transfer the standard in the MRBench to points
- run the python main_code/transfer_MRB2point.py 
### extract points from the json files
#### to clean the data , run the  python main_code/clean_result.py, this will produce the cleaned_evaluation_result.json

# compare the file with the Benchmark and count the correlation
- run python main_code/count_correlation.py
### transfer the standard in the MRBench to points
#### run the python main_code/transfer_MRB2point.py 

### compare the file with the Benchmark and count the correlation
#### run python main_code/count_correlation.py

### result from (1)

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
LLAMA       19  [5, 14, 18, 37, 56, 67, 76, 89, 92, 94, 98, 107, 124, 141, 149, 152, 162, 166, 180]
MISTRAL     14  [24, 35, 37, 43, 78, 91, 100, 114, 127, 128, 151, 166, 168, 169]

## (2) First count the point according to the golden standard and compare to the result in the paper

### run python main_code/counting_final_result.py and get three compare_paper_my/ neg1_item_by_model_dimension/ damr_by_model_dimension.csv
### result from (2)

| model   | Mistake Identification_paper | Mistake Location_paper | Revealing the Answer_paper | Providing Guidance_paper | Actionability_paper | Coherence_paper | Tutor Tone_paper | Human-likeness_paper | Actionability_my | Coherence_my | Human-likeness_my | Mistake Identification_my | Mistake Location_my | Providing Guidance_my | Revealing the Answer_my | Tutor Tone_my | Mistake Identification_diff(my-paper) | Mistake Location_diff(my-paper) | Revealing the Answer_diff(my-paper) | Providing Guidance_diff(my-paper) | Actionability_diff(my-paper) | Coherence_diff(my-paper) | Tutor Tone_diff(my-paper) | Human-likeness_diff(my-paper) |
|---------|------------------------------|------------------------|----------------------------|--------------------------|---------------------|-----------------|------------------|-----------------------|------------------|--------------|-------------------|--------------------------|--------------------|-----------------------|-------------------------|---------------|--------------------------------------|--------------------------------|-------------------------------------|----------------------------------|-----------------------------|---------------------------|----------------------------|-------------------------------|
| Llama   | 80.21                        | 54.69                  | 73.96                      | 45.31                    | 42.71               | 80.73           | 19.79            | 93.75                 | 96.53            | 98.28        | 98.84             | 83.24                    | 84.39              | 83.82                 | 27.75                   | 20.81         | 3.03                                 | 29.70                          | -46.21                             | 38.51                            | 53.82                       | 17.55                     | 1.02                          | 5.09                          |
| Mistral | 93.23                        | 73.44                  | 86.46                      | 63.54                    | 70.31               | 86.98           | 15.10            | 95.31                 | 55.06            | 60.34        | 73.03             | 43.82                    | 43.82              | 44.38                 | 53.93                   | 22.47         | -49.41                               | -29.62                         | -32.53                             | -19.16                           | -15.25                      | -26.64                    | 7.37                          | -22.28                        |
