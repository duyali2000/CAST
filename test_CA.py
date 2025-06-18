import json
import os
import re
import ast
from tqdm import tqdm
import signal
import argparse
import jsonlines
import numpy as np
import eventlet  # 导入eventlet这个模块

eventlet.monkey_patch()  # 必须加这条代码

def TimeoutHandler(signum, frame):
    raise Exception("timeout")

def evaluate_scripts(model, src_lang, dst_lang, k, suffix, timeout=5):
    script_file_path = f"./cleaned_data/{model}/test_scripts{suffix}/{src_lang}_{dst_lang}"
    testable_file_lst = os.listdir(script_file_path)

    gold_dir = f"./cleaned_data/transcoder_evaluation_gfg/{dst_lang}/"
    print("num: ", len(testable_file_lst))
    log_details = []
    all_log = []
    all_pass_rate_log = []
    num = 0
    for file in tqdm(testable_file_lst):

        sample_lst = os.listdir(f"{script_file_path}/{file}")
        topk = min(len(sample_lst), k)
        log_lst = []
        log_pass_rate_lst = []

        # num += 1
        # if (num == 397):
        #     log_lst.append(False)
        #     log_pass_rate_lst.append(0)
        #     continue



        for id in range(topk):
            # linux 可以用signal， windows环境建议用eventlet设置超时
            #signal.signal(signal.SIGALRM, TimeoutHandler)
            #signal.alarm(timeout)
            try:
                with eventlet.Timeout(5, True):  # 设置超时时间为2秒
                    if dst_lang == "python":
                        gold_file = gold_dir + file + ".py"
                        #result = os.popen(f"python {script_file_path}/{file}/sample_{id}.py")
                        # 读取 script_file 的内容
                        with open(f"{script_file_path}/{file}/sample_{id}.py", 'r') as script_f:
                            script_content = script_f.read()
                            func_name = locate_function_name_py(script_content)[0]
                            script_content = script_content.replace(func_name, "f_filled", 1)
                        # 读取 gold_file 的内容
                        with open(gold_file, 'r') as gold_f:
                            gold_content = gold_f.read()
                        # 替换 gold_file 中的 //TOFILL
                        gold_content_replaced = gold_content.replace('#TOFILL', script_content)
                        # 保存替换后的内容到 gold_file
                        new_gold = gold_dir + file + "_tmp.py"
                        with open(new_gold, 'w') as gold_f:
                            gold_f.write(gold_content_replaced)
                        result = os.popen(f"python {new_gold}")
                        # 此时打开的a是一个对象，如果直接打印的话是对象内存地址
                    elif dst_lang == "java":
                        gold_file = gold_dir + file + ".java"
                        #result = os.popen(f"java --module-path $PATH_TO_FX --add-modules javafx.controls {script_file_path}/{file}/sample_{id}.java")
                        # 读取 script_file 的内容
                        with open(f"{script_file_path}/{file}/sample_{id}.java", 'r') as script_f:
                            script_content0 = script_f.read()
                            func_name = locate_function_name_java(script_content0)[0]
                            script_content = script_content0.replace(func_name, "f_filled", 1)
                        # 读取 gold_file 的内容
                        with open(gold_file, 'r') as gold_f:
                            gold_content = gold_f.read()
                        # 替换 gold_file 中的 //TOFILL
                        gold_content_replaced = gold_content.replace('//TOFILL', script_content)
                        # 保存替换后的内容到 gold_file
                        new_gold = gold_dir + file + "_tmp.java"
                        with open(new_gold, 'w') as gold_p:
                            gold_p.write(gold_content_replaced)
                        result = os.popen(f"java --module-path %PATH_TO_FX% --add-modules javafx.controls {new_gold}")
                        #result = os.popen(f"javac {new_gold}")
                    elif dst_lang == "cpp":
                        gold_file = gold_dir + file + ".cpp"
                        # os.system(f"g++ {script_file_path}/{file}/sample_{id}.cpp -o {script_file_path}/{file}/sample_{id}")
                        # result = os.popen(f"{script_file_path}/{file}/sample_{id}")
                        # 读取 script_file 的内容
                        with open(f"{script_file_path}/{file}/sample_{id}.cpp", 'r') as script_f:
                            script_content = script_f.read()
                            func_name = locate_function_name_cpp(script_content)[0]
                            script_content = script_content.replace(func_name, "f_filled", 1)
                        # 读取 gold_file 的内容
                        with open(gold_file, 'r') as gold_f:
                            gold_content = gold_f.read()
                        # 替换 gold_file 中的 //TOFILL
                        gold_content_replaced = gold_content.replace('//TOFILL', script_content)
                        # 保存替换后的内容到 gold_file
                        new_gold = gold_dir + file + "_tmp.cpp"
                        with open(new_gold, 'w') as gold_f:
                            gold_f.write(gold_content_replaced)
                        os.system(f"g++ {new_gold} -o {script_file_path}/{file}/sample_{id}")
                        result = os.popen(f"{script_file_path}/{file}/sample_{id}")
                    else:
                        assert False, "unknown dst_lang!"

                    log = result.read()
                    print(log)
                    # 要用read（）方法读取后才是文本对象
                    log_ret = re.findall(r"#Results:\s?\d+,\s?\d+\n?", log)
                    if len(log_ret) > 0:
                        act, tal = log_ret[0].replace("#Results:", "").split(",")
                        # PASS@K
                        if int(act.strip()) == int(tal.strip()):
                            log_lst.append(True)
                        else:
                            log_lst.append(False)
                        # PASS RATE
                        log_pass_rate_lst.append(int(act.strip()) / int(tal.strip()))
                    else:
                        log_lst.append(False)
                        log_pass_rate_lst.append(0)
                        print(gold_content_replaced)
                    #signal.alarm(0)
            except eventlet.timeout.Timeout:
                log_lst.append(False)
                log_pass_rate_lst.append(0)
                # signal.alarm(0)
                continue
            except Exception as ret:
                print("EXCEPTION:", ret)
                log_lst.append(False)
                log_pass_rate_lst.append(0)
                #signal.alarm(0)
                continue
        log_details.append({"id": file, "ret": log_lst, "rate": log_pass_rate_lst})
        if True in log_lst:
            # exist at least one true candidate
            all_log.append(True)
        else:
            all_log.append(False)
        all_pass_rate_log.append(np.max(log_pass_rate_lst))
    print(f"computational acc: {all_log.count(True) / len(all_log)}, Correct_num: {all_log.count(True)}, Total: {len(all_log)}")
    print(f"AVG PASS RATE: {np.mean(all_pass_rate_log)}")
    log_details.append({f"computational acc: {all_log.count(True) / len(all_log)}, Correct_num: {all_log.count(True)}, Total: {len(all_log)}, AVG PASS RATE: {np.mean(all_pass_rate_log)}"})
    with jsonlines.open(f"./log_details_{model}_{src_lang}_{dst_lang}{suffix}.jsonl", "w") as fw:
        fw.write_all(log_details)



def locate_function_name_py(code):
    pattern = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)")
    method_info = pattern.findall(code)
    try:
        var_lst = []
        pattern_pa = re.compile(r"(\w+)\s*")
        if method_info:
            params = method_info[0][1]
            for param in params.split(","):
                match = pattern_pa.match(param.strip())
                if match:
                    var_lst.append(match.group(1))
            return method_info[0][0], var_lst
        else:
            return None, None
    except Exception as e:
        return None, None



def locate_function_name_java(code):
    pattern = re.compile(r"(public|private|protected)?\s?(static)?\s?(\w+|\w+\[\])\s(\w+)\s?\(\s?(\w+.*\w*)?\s?\)")
    method_info = pattern.findall(code)
    try:
        pattern_pa1 = re.compile(r"\w+\s(\w+)")
        pattern_pa2 = re.compile(r"\w+\s?[\[\s?\]]+\s(\w+)")
        var_lst = []
        for i in method_info[0][-1].split(","):
            if len(pattern_pa1.findall(i)) > 0:
                var_lst.append(pattern_pa1.findall(i)[0])
            elif len(pattern_pa2.findall(i)) > 0:
                var_lst.append(pattern_pa2.findall(i)[0])
        #return method_info[0][3], method_info[0][2], var_lst
        return method_info[0][3],method_info[0][2], var_lst
    except Exception as e:
        return None, None, None

def locate_function_name_cpp(code):
    code4func = re.sub(r"//.+?\n", "", code)
    pattern = re.compile(r"([\w\s\*]+)\s(\w+)\s?\(\s?(\w+.*\w*)?\s?\)")
    method_info = pattern.findall(code4func)
    try:
        var_lst = []
        # int a, const std::vector<std::vector<double>>& m, vector<vector<int>> mat
        pattern_pa1 = re.compile(r"\w+\s(\w+)$")
        # int * xp; int & xp;
        pattern_pa2 = re.compile(r"\w+\s?[\*\&]{1}\s?(\w+)")
        # int arr [ ]
        pattern_pa3 = re.compile(r"\w+\s?(\w+)\s?\[\s?\]")
        # print("VARS: ", method_info[0][-1].split(","))
        for i in method_info[0][-1].split(","):
            # for-> unsigned int/long.. var;
            i = i.replace("unsigned", "").strip()
            if len(pattern_pa2.findall(i)) > 0:
                var_lst.append(pattern_pa2.findall(i)[0])
            elif len(pattern_pa3.findall(i)) > 0:
                var_lst.append(pattern_pa3.findall(i)[0])
            elif len(pattern_pa1.findall(i)) > 0:
                var_lst.append(pattern_pa1.findall(i)[0])
        # print("var_lst: ", var_lst)
        return method_info[0][1], method_info[0][0], var_lst
    except Exception as e:
        return None, None, None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, help="source language", default="python")
    parser.add_argument("--dst_lang", type=str, help="target language", default="java")
    parser.add_argument("--k", type=int, help="pass@k", default=1)
    parser.add_argument("--timeout", type=int, help="timeout", default=1)
    parser.add_argument("--model", type=str, help="select an llm", default="gpt3_5")
    parser.add_argument("--suffix", type=str, help="suffix of experiments", default="_zero_shot")
    args = parser.parse_args()
    model = args.model
    src_lang = args.src_lang
    dst_lang = args.dst_lang
    evaluate_scripts(model, src_lang, dst_lang, k=args.k, suffix=args.suffix, timeout=args.timeout)
# python test_CA.py --model llama7b --src_lang cpp --dst_lang java --k 1 --timeout 5 --suffix _w_0cases_2round