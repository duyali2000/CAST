import os
import subprocess
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE
import argparse

JAVA_ROOT = "/home/duyl/hallucination/sourcecode/tools/jdk-11.0.27/bin"


def main(args):
    print('testing translations')
    dataset = 'avatar'
    # translation_dir = f"output/{args.model}/{dataset}/{args.source_lang}/{args.target_lang}"
    translation_dir = f"/data2/duyl/dataset/avatar/cleaned_data/{args.model}/test_scripts_w_syntax_{args.shots}_shot_{args.selection}_selection/{args.source_lang}_{args.target_lang}"
    test_dir = f"/data2/duyl/dataset/{dataset}/{args.source_lang}/TestCases"
    os.makedirs(args.report_dir, exist_ok=True)
    files = [f for f in os.listdir(translation_dir) if f.split(".")[-1] in ["py", "java", "c", "cpp", "go"]]

    if args.model == "gpt4o" and args.shots == 0 and args.selection == "none":
        test_files = [f.split(".")[0] for f in os.listdir(f"/data2/duyl/dataset/avatar/{args.source_lang}/Code") if f.split(".")[-1] in ["py", "java", "c", "cpp", "go"]]
        files = [f for f in files if f.split(".")[0] in test_files]

    compile_failed = []
    test_passed =[]
    test_failed =[]
    test_failed_details = []
    runtime_failed = []
    runtime_failed_details= []
    infinite_loop = []
    compile_success = []

    if args.target_lang =="python":
        for i in range(0,len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("python3 -m py_compile "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)
                compile_success.append(files[i])
                tests_passed = 0
                for j in range(1000):

                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                        
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen(['python3', translation_dir+"/"+ files[i]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)    

                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=10)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try: # if stdout is already decoded as String, then pass
                            stdout = stdout.decode()
                        except:
                            pass

                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))
                
            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang =="java":
        for i in range(0,len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run(f"{JAVA_ROOT}/javac "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)
                compile_success.append(files[i])
                tests_passed = 0
                for j in range(1000):

                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                        
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen([f'{JAVA_ROOT}/java', files[i].split(".")[0]], cwd=translation_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)

                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=10)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try:
                            stdout = stdout.decode()
                        except:
                            pass

                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))

            except Exception as e:
                compile_failed.append(files[i])

        #remove all .class files generated
        dir_files = os.listdir(translation_dir)
        for fil in dir_files:
            if ".class" in fil: os.remove(translation_dir +"/"+ fil)

    elif args.target_lang == "c":
        for i in range(0,len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("gcc "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=10)

                tests_passed = 0
                for j in range(1000):

                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                        
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen(['./a.out'], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)

                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try:
                            stdout = stdout.decode()
                        except:
                            pass

                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))

            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang == "cpp":
        for i in range(0,len(files)):

            try:
                print('Filename: ', files[i])
                subprocess.run("g++ -o exec_output -std=c++11 " + translation_dir + "/" + files[i], check=True, capture_output=True, shell=True, timeout=10)
                compile_success.append(files[i])
                
                tests_passed = 0
                for j in range(1000):

                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                        
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen(['./exec_output'], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
                    
                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=10)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try:
                            stdout = stdout.decode()
                        except:
                            pass

                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))
                
            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang == "go":
        for i in range(0,len(files)):
            
            try:
                print('Filename: ', files[i])
                subprocess.run("go build "+ translation_dir + "/" + files[i], check=True, capture_output=True, shell=True, timeout=30)
                
                tests_passed = 0
                for j in range(1000):

                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                        
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen(["./"+files[i].split(".")[0]], cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, stderr=PIPE)    

                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try:
                            stdout = stdout.decode()
                        except:
                            pass

                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))

            except Exception as e:
                compile_failed.append(files[i])

    else:
        print("language:{} is not yet supported. select from the following languages[Python,Java,C++,C,Go]".format(args.target_lang))
        return

    test_failed = list(set(test_failed))
    test_failed_details = list(set(test_failed_details))
    runtime_failed = list(set(runtime_failed))
    runtime_failed_details = list(set(runtime_failed_details))
    compile_failed = list(set(compile_failed))
    infinite_loop = list(set(infinite_loop))
    test_passed = list(set(test_passed))

    # To avoid the total sum is higher than 100%, if an instance is in infinite_loop and test_failed at the same time, then it will be counted as test_failed
    for instance in infinite_loop[:]:
        if instance in test_failed:
            infinite_loop.remove(instance)

    txt_fp = Path(args.report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+".txt")
    with open(txt_fp, "w", encoding="utf-8") as report:
        report.writelines("Total Instances: {}\n\n".format(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop)))
        report.writelines("Total Correct: {}\n".format(len(test_passed)))
        report.writelines("Total Runtime Failed: {}\n".format(len(runtime_failed)))
        report.writelines("Total Compilation Failed: {}\n".format(len(compile_failed)))
        report.writelines("Total Test Failed: {}\n".format(len(test_failed)))
        report.writelines("Total Infinite Loop: {}\n\n".format(len(infinite_loop)))
        report.writelines("Accuracy: {}\n".format((len(test_passed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
        report.writelines("Runtime Rate: {}\n".format((len(runtime_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
        report.writelines("Compilation Rate: {}\n".format((len(compile_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
        report.writelines("Test Failed Rate: {}\n".format((len(test_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
        report.writelines("Infinite Loop Rate: {}\n\n".format((len(infinite_loop)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
        report.writelines("=================================================================================================\n")
        report.writelines("Failed Test Files: {} \n".format(test_failed))
        report.writelines("Failed Test Details: {} \n".format(test_failed_details))
        report.writelines("=================================================================================================\n")
        report.writelines("Runtime Error Files: {} \n".format(runtime_failed))
        report.writelines("Runtime Error Details: {} \n".format(runtime_failed_details))
        report.writelines("=================================================================================================\n")
        report.writelines("Compilation Error Files: {} \n".format(compile_failed))
        report.writelines("=================================================================================================\n")    
        report.writelines("Compilation Error Files: {} \n".format(compile_success))
        report.writelines("=================================================================================================\n")  
        report.writelines("Infinite Loop Files: {} \n".format(infinite_loop))
        report.writelines("=================================================================================================\n")
        


    print(compile_success)
    print("Total Instances: {}\n\n".format(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop)))
    print("Total Correct: {}\n".format(len(test_passed)))
    print("Total Runtime Failed: {}\n".format(len(runtime_failed)))
    print("Total Compilation Failed: {}\n".format(len(compile_failed)))
    print("Total Test Failed: {}\n".format(len(test_failed)))
    print("Total Infinite Loop: {}\n\n".format(len(infinite_loop)))
    print("Accuracy: {}\n".format((len(test_passed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
    print("Runtime Rate: {}\n".format((len(runtime_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
    print("Compilation Rate: {}\n".format((len(compile_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
    print("Test Failed Rate: {}\n".format((len(test_failed)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))
    print("Infinite Loop Rate: {}\n\n".format((len(infinite_loop)/(len(test_passed)+len(compile_failed)+len(runtime_failed)+len(test_failed)+len(infinite_loop))) * 100))


    df = pd.DataFrame(columns=['Source Language', 'Target Language', 'Filename', 'BugType', 'RootCause', 'Impact', 'Comments'])
    index = 0
    for i in range(0, len(compile_failed)):
        list_row = [args.source_lang, args.target_lang, compile_failed[i], "", "", "Compilation Error", ""]
        df.loc[i] = list_row
        index+=1
    for i in range(0, len(runtime_failed)):
        list_row = [args.source_lang, args.target_lang, runtime_failed[i], "", "", "Runtime Error", ""]
        df.loc[index] = list_row
        index+=1 
    for i in range(0, len(test_failed)):
        list_row = [args.source_lang, args.target_lang, test_failed[i], "", "", "Test Failed", ""]
        df.loc[index] = list_row
        index+=1
    
    excel_fp = Path(args.report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+".xlsx")
    df.to_excel(excel_fp, sheet_name='Sheet1')

    ordered_unsuccessful_fp = Path(args.report_dir).joinpath(f"{args.model}_{dataset}_compileReport_from_"+str(args.source_lang)+"_to_"+str(args.target_lang)+"_ordered_unsuccessful.txt")
    with open(ordered_unsuccessful_fp, 'w') as f:
        for unsuccessful_instance in compile_failed + runtime_failed + test_failed + infinite_loop:
            f.write(f"{unsuccessful_instance}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='execute avatar tests')
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [Python,Java,C,C++,Go]', required=True, type=str)
    parser.add_argument("--selection", type=str, default="coverage")
    parser.add_argument("--shots", type=int, help="one shot or not", default=1)
    parser.add_argument('--model', help='model to use for code translation.', required=True, type=str)
    parser.add_argument('--report_dir', help='path to directory to store report', required=True, type=str)
    args = parser.parse_args()

    main(args)


# python compile_avatar.py --source_lang java --target_lang python --model qwen7b --shots 5 --selection random --report_dir ./report