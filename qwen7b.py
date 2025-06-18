import re
import requests
# import openai
import json
import jsonlines
from tqdm import tqdm
from cleaned_data.templates.examples_inp import example_cpp, valid_inputs_cpp, \
    example_java, valid_inputs_java, example_python, valid_inputs_python
from cleaned_data.templates.examples_trans import \
    example_code_java, example_code_cpp, example_code_python, \
    example_test_cases_java, example_test_cases_cpp, example_test_cases_python
from cleaned_data.templates.example_refine import python_refine_example2_1, python_refine_example2_2, \
    cpp_refine_example2_1, cpp_refine_example2_2, java_refine_example2_1, java_refine_example2_2
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
)  # for exponential backoff
import argparse

# 最少等待1秒，最多10秒
#@retry(wait=wait_random_exponential(min=1, max=5), retry=retry_if_exception_type())
def collect_one(prompt, api_key, sample_num=20):

    url = api_key

    content = ''
    for i in prompt:
        content += i['content']
    print(content)

    params = {
        "model": "qwen2:7b",
        "prompt": content,
        "stream": False
    }

    response = requests.post(url, data=json.dumps(params))

    print(response.text)

    candidates = [response.text]
    return candidates


def prompt_trans(src_lang, dst_lang, item):
    return {"role": "user", "content": f"Given the {src_lang} code:\n{item}\nPlease translate the above {src_lang} code to {dst_lang} code, and use END_OF_CASE to finish your answer."}

def prompt_trans_one_shot(src_lang, dst_lang, item):
    example_dst_code = eval(f"example_code_{dst_lang}")
    example_src_code = eval(f"example_code_{src_lang}")
    content = f"Given {src_lang} code:\n{example_src_code}\nTranslate given {src_lang} code to {dst_lang} code, " \
              f"and use END_OF_CASE to finish your answer.\n{example_dst_code}\nEND_OF_CASE\n"
    target = f"Given {src_lang} code:\n{item}\nTranslate given {src_lang} code to {dst_lang} code, " \
              f" only output the code and use END_OF_CASE to finish your answer.\n"
    content = content + target
    return {"role": "user", "content": content}



def main(data_path, src_lang, dst_lang, obj, sample_num, api_key, out_path, test_case_num, feedback_file, org_sol_file, start):
    data = []
    sample_ids = []
    with open(data_path, encoding="utf-8") as fr:
        for line in fr.readlines():
            line = json.loads(line)
            data.append(line[src_lang])
            sample_ids.append(line['id'])

    prefix = [{"role": "system", "content": "You are a professional developer proficient in java, python, and cpp.\n"}]
    count = 0
    for id, item in tqdm(zip(sample_ids, data), total=len(sample_ids)):
        count += 1
        if count < start:
            continue
        if obj == TRANS:
            target = [prompt_trans(src_lang, dst_lang, item)]
        elif obj == TRANS_ONE_SHOT:
            target = [prompt_trans_one_shot(src_lang, dst_lang, item)]
        else:
            assert False, "no such objective!"
        prompt = prefix + target
        with jsonlines.open(out_path, "a") as fw:
            fw.write({"id": id, dst_lang: collect_one(prompt, api_key, sample_num=sample_num)})


if __name__ == "__main__":
    GEN_VAL_INP = 0
    TRANS = 1
    TRANS_ONE_SHOT = 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, help="enter your api key", default="http://210.28.135.196:11434/api/generate")
    parser.add_argument("--src_lang", type=str, help="source language", default="java")
    parser.add_argument("--dst_lang", type=str, help="target language", default="python")
    parser.add_argument("--k", type=int, help="sampling number", default=10)
    parser.add_argument("--start", type=int, help="start index", default=1)
    parser.add_argument("--shots", type=int, help="one shot or not", default=1)
    parser.add_argument("--round", type=int, help="number of round", default=2)
    parser.add_argument("--test_case_num", type=int, help="num of test cases", default=3)
    parser.add_argument("--obj", type=int, help="select an objective - 0: gen_val_inp, 2: trans, 3: trans_w_cases, 4: refine", default=0)
    args = parser.parse_args()



    API_KEY = args.apikey
    src_lang = args.src_lang
    dst_lang = args.dst_lang
    obj = args.obj
    sample_num = args.k
    _round = args.round
    test_case_num = args.test_case_num  # todo only be activated when obj == TRANS_W_CASES
    test_file = f"./cleaned_data/testable_samples.jsonl"
    # 用于第N轮refine的feedbacks
    feedback_file = f"./cleaned_data/qwen7b/feedbacks/testable_{src_lang}_{dst_lang}_w_{test_case_num}cases_{_round}round.jsonl"
    # 第N-1轮作为原始结果
    org_sol_file = f"./cleaned_data/qwen7b/post_processed_w_{test_case_num}cases_{_round}round/testable_{src_lang}_{dst_lang}_w_{test_case_num}cases_{_round}round.jsonl"
    # todo not adapt for gen_val_inp yet
    # 第N轮refine输出
    out_path = f"./cleaned_data/qwen7b/testable_{src_lang}_{dst_lang}_w_{test_case_num}cases_{_round}round.jsonl"
    # TODO: init: start=1, restart: start=stop_count+1
    main(test_file, src_lang, dst_lang, obj, sample_num, API_KEY, out_path, test_case_num, feedback_file, org_sol_file, start=1)



# python qwen7b.py --src_lang java --dst_lang cpp --obj 2 --k 10 --test_case_num 0
# python qwen7b.py --src_lang java --dst_lang cpp --obj 0 --k 10 --test_case_num 20