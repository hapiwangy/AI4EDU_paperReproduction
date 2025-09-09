# this Package is used to establish the prompts based on different cases
def MathDial_Prompt(org_prompt, streamdata, org=True):
    # if org == True: use the original mode, else use the specific mode
    if org:
        return org_prompt.format(history=streamdata['conversation_history'])
    else:
        org_prompt = org_prompt[0]
        return [
            {"role":"system", "content":org_prompt['system']},
            {"role":"user", "content":org_prompt['user'].format(history=streamdata['conversation_history'])}
            ]
def Bridge_Prompt(org_prompt, streamdata, org=True):
    if org:
        return org_prompt.format(topic= streamdata['Topic'], history=streamdata['conversation_history'])
    else:
        org_prompt = org_prompt[0]
        return [
            {"role":"system", "content":org_prompt['system']},
            {"role":"user", "content":org_prompt['user'].format(topic= streamdata['Topic'], history=streamdata['conversation_history'])}
            ]
        # return "<|begin_of_text|>" + org_prompt.format(topic= streamdata['Topic'], history=streamdata['conversation_history'])

def evaluation_prompt(org_prompt, streamdata):
    from string import Template
    return Template(org_prompt).substitute(history=streamdata['conversation_history'], response=streamdata['result'])

def safe_cut_at_first_heading(text: str) -> str:
    return text.split("###", 1)[0].strip()

def cutting_out_answer(result):
    # define standard
    definitions = {"mistake_identification": "Has the tutor identified a mistake in a student’s response?",
    "mistake_location": "Does the tutor’s response accurately point to a genuine mistake and its location?",
    "revealing_answer": "Does the tutor reveal the final answer (whether correct or not)?",
    "providing_guidance": "Does the tutor offer correct and relevant guidance, such as an explanation, elaboration, hint,examples, and so on?",
    "coherent": "Is the tutor’s response logically consistent with the student’s previous response?",
    "actionability": "Is it clear from the tutor’s feedback what the student should do next?",
    "tutor_tone": "Is the tutor’s response encouraging, neutral, or offensive?",
    "humanness": "Does the tutor’s response sound natural, rather than robotic or artificial?"}
    definitions = tuple(definitions.keys())
    # define point 2 rate
    point2rate = {
        "mistake_identification_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "mistake_location_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
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
        "coherent_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "actionability_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        },
        "tutor_tone_rubric": {
            1: "Encouraging",
            2: "Neutral",
            3: "Offensive"
        },
        "humanness_rubric": {
            1: "Yes",
            2: "To some extent",
            3: "No"
        }
    }
    temp = {}
    for d in definitions:
        target_word = f"{d} = "
        score = result.find(target_word)
        try:
            grade = int(result[score + len(target_word)])
        except:
            grade = -1
        temp[f"{d}_point"] = grade
        if grade in (1, 2, 3):
            temp[d] = point2rate[f"{d}_rubric"][grade]
        else:
            temp[d] = "grad not exist"
    return temp
if __name__ == '__main__':
    import json
    with open(f"/u/cwang34/AI+EDU project/MRBench_V1/train.json", "r", encoding="utf-8") as fp:
        json_data = json.load(fp)
    with open("AI+EDU project/MRBench_V1/llama_prompt_MathDial.json", "r") as fp:
        MathPrompt = json.load(fp)
    print(json_data[0])
    print(MathDial_Prompt(MathPrompt, json_data[0]))

