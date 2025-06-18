import copy
import re
import requests
# import openai
import json
import jsonlines
from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_random_exponential,
)  # for exponential backoff
import argparse
import os
from syntax_generator import get_type_list, filter_syntax
from prompt_template import prompt_trans, prompt_trans_one_shot, prompt_trans_few_shot, prompt_trans_one_shot_syntax, prompt_trans_one_shot_code
from incontext import fix_incontext_syntax
import torch
from transformers import RobertaTokenizer, RobertaModel
# 最少等待1秒，最多10秒
@retry(wait=wait_random_exponential(min=1, max=5), retry=retry_if_exception_type())
def collect_one(prompt, api_key, modelname, sample_num=20):

    url = api_key

    content = ''
    for i in prompt:
        content += i['content']
    #print(content)

    params = {
        "model": f"{modelname}",
        "prompt": content,
        "stream": False
    }

    response = requests.post(url, data=json.dumps(params))

    #print(response.text)

    candidates = [response.text]
    return candidates


#GPT-4
# 最少等待1秒，最多10秒
# @retry(wait=wait_random_exponential(min=5, max=10), retry=retry_if_exception_type())
# def collect_one(prompt, api_key, sample_num=20):
#     # API_SECRET_KEY = "xxxxxx";
#     # BASE_URL = "https://api.zhizengzeng.com/v1/"
#
#     API_SECRET_KEY = "sk-zk238a51b86ab218c5f6277b7ac2017ed7865ee8fbc93104"
#     BASE_URL = "https://api.zhizengzeng.com/v1/"
#
#     content = ''
#     for i in prompt:
#         content += i['content']
#     print(content)
#
#     client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
#     resp = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": content}
#         ]
#     )
#     #print(resp.choices[0].message.content)
#     print(resp)
#
#     candidates = [resp.choices[0].message.content]
#     return candidates



def get_code_embedding(codes):

    tokenizer = RobertaTokenizer.from_pretrained('/data2/duyl/pretrainmodels/codebert')
    encoder = RobertaModel.from_pretrained('/data2/duyl/pretrainmodels/codebert').to('cuda')
    # 将代码进行tokenize并获取相应的输入ID
    codeimbeddings = {}
    for code in codes:
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')
        # 获取模型的最后一层输出并提取[CLS]嵌入
        with torch.no_grad():
            outputs = encoder(**inputs)
            # 使用[CLS]标记的嵌入表示整个句子的语义
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
        codeimbeddings[code] = cls_embedding
    return codeimbeddings



def main(data_path, src_lang, dst_lang, obj, sample_num, api_key, out_path, shots, mname, outdir, round, start):
    # load database when selection
    databank = {}
    correctfiles = []
    allsourcecode = []
    if (obj == TRANS_FEW_SHOT):

        file_path = f"{outdir}{src_lang}_{dst_lang}_round1_results.jsonl"
        with open(file_path, encoding="utf-8") as f0:
            for line in f0.readlines():
                line = json.loads(line)
                correctfiles.append(line["id"])
                databank[line[src_lang]] = line[dst_lang]
                allsourcecode.append(line[src_lang])


    data = []
    sample_ids = []
    # with open(data_path, encoding="utf-8") as fr:
    #     for line in fr.readlines():
    #         line = json.loads(line)
    #         data.append(line[src_lang])
    #         sample_ids.append(line['id'])

    # 遍历文件夹下所有文件
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)

        # 跳过子文件夹，仅处理文件
        if not os.path.isfile(file_path):
            continue

        # 提取sample_id（使用文件名作为标识）
        sample_id = filename  # 若需要去除扩展名，可使用os.path.splitext(filename)[0]

        # 读取文件内容
        with open(file_path, encoding="utf-8") as fr:
            file_content = fr.read()  # 按文件类型选择读取方式（文本/JSON等）
            data_line = file_content.strip()

            data.append(data_line)
            sample_ids.append(sample_id)
            allsourcecode.append(data_line)


    prefix = [{"role": "system", "content": "You are a professional developer proficient in java, python, and cpp.\n"}]
    count = 0

    shotnum = []

    codeimbeddings = get_code_embedding(allsourcecode)


    for id, item in tqdm(zip(sample_ids, data), total=len(sample_ids)):
        count += 1
        if count < start:
            continue

        if (id in correctfiles and obj == TRANS_FEW_SHOT):
            continue

        for i in range(5):
            if obj == TRANS:
                target = [prompt_trans(src_lang, dst_lang, item)]
            elif obj == TRANS_ONE_SHOT:
                target = [prompt_trans_one_shot(src_lang, dst_lang, item)]
            elif obj == TRANS_FEW_SHOT:
                if(args.selection == "autocoverage"):
                    tmp, itemshots = prompt_trans_few_shot(src_lang, dst_lang, databank, codeimbeddings, item, outdir, args.selection, round, shots, args.threshold)
                    target = [tmp]
                else:
                    target = [prompt_trans_few_shot(src_lang, dst_lang, databank, codeimbeddings, item, outdir, args.selection, round, shots, args.threshold)]
            else:
                assert False, "no such objective!"
            prompt = prefix + target
            candidates = collect_one(prompt, api_key, modelname=mname, sample_num=sample_num)
            print(candidates)
            pred_code = json.loads(candidates[0])['response'].split("END")[0]
            target_signal = filter_syntax(pred_code,
                                          dst_lang)  # filter by syntax corresponding relationship, and target parser

            if (target_signal):
                # pred_syntax = get_type_list(pred_code, dst_lang)
                #
                # if(pred_syntax == dst_syntax):
                with jsonlines.open(out_path, "a") as fw:
                    fw.write({"id": id, dst_lang: candidates})
                if (args.selection == "autocoverage"):
                    shotnum.append(itemshots)
                    #print(itemshots)
                break

            if(i == 4):
                with jsonlines.open(out_path, "a") as fw:
                    fw.write({"id": id, dst_lang: candidates})
                if (args.selection == "autocoverage"):
                    shotnum.append(itemshots)
    # print(shotnum)
    # average_shotnum = sum(shotnum) / len(shotnum) if shotnum else 0
    # print(average_shotnum)
    # with open("shortnum.txt", 'a+') as file:
    #     file.write(f"{src_lang},{dst_lang},{args.threshold},{average_shotnum}\n")

if __name__ == "__main__":
    GEN_VAL_INP = 0
    TRANS = 1
    TRANS_ONE_SHOT = 2
    TRANS_FEW_SHOT = 3
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--apikey", type=str, help="enter your api key", default="http://210.28.135.73:11434/api/generate")
    parser.add_argument("--src_lang", type=str, help="source language", default="java")
    parser.add_argument("--dst_lang", type=str, help="target language", default="python")
    parser.add_argument("--model_name", type=str, help="llm", default="gemma2:27b")
    parser.add_argument("--shortname", type=str, help="llm")
    parser.add_argument("--k", type=int, help="sampling number", default=10)
    parser.add_argument("--start", type=int, help="start index", default=1)
    parser.add_argument("--shots", type=int, help="one shot or not", default=1)
    parser.add_argument("--round", type=int, help="number of round", default=0)
    parser.add_argument("--obj", type=int, help="select an objective - 0: gen_val_inp, 2: trans, 3: trans_w_cases, 4: refine", default=3)
    parser.add_argument("--selection", type=str, default="coverage")
    parser.add_argument("--threshold", type=float, default="0.92")
    args = parser.parse_args()



    API_KEY = args.apikey
    src_lang = args.src_lang
    dst_lang = args.dst_lang
    obj = args.obj
    sample_num = args.k
    _round = args.round

    startnum = 0

    test_file = f"/home/duyl/grammartran/avatar/{src_lang}/Code"
    out_dir = f"/home/duyl/grammartran/avatar/cleaned_data/{args.shortname}/"
    #out_path = out_dir + f"testable_{src_lang}_{dst_lang}_w_syntax_fix_{args.shots}_shot_{args.selection}_selection_threshold_{args.threshold}.jsonl"
    out_path = out_dir + f"testable_{src_lang}_{dst_lang}_w_syntax_fix_{args.shots}_shot_{args.selection}_selection.jsonl"
    # TODO: init: start=1, restart: start=stop_count+1
    main(test_file, src_lang, dst_lang, obj, sample_num, API_KEY, out_path, shots=args.shots, mname=args.model_name, outdir=out_dir, round=args.round, start=startnum)



# python method_few_shot.py --src_lang cpp --dst_lang python --obj 3 --k 10  --model_name qwen2:7b --shots 5 --round 2 --selection similarity --shortname qwen7b
# python method_few_shot.py --src_lang cpp --dst_lang python --obj 3 --k 10  --model_name gpt4 --shots 5 --round 2 --selection coverage --shortname gpt4