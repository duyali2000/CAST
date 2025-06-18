import json
import re
import os


# Function to extract and save cpp code
def extract_and_save(json_line, model, source, target):
    try:
        data = json.loads(json_line)
        id = data['id']
        cpp_ = data[target][0]
        cpp_code = json.loads(cpp_)['response']
        cpp_code = cpp_code.split("END")[0]

        print(cpp_code)
        script_file_path = f"/home/duyl/grammartran/avatar/cleaned_data/{model}/test_scripts_w_syntax_{shots}_shot_{stra}_selection/{source}_{target}"
        # Create directory if it doesn't exist
        if not os.path.exists(f"{script_file_path}"):
            os.makedirs(f"{script_file_path}")

        filename = id.split(".")[0]
        # Save cpp code to file
        # Save cpp code to file
        if(target == "java"):
            if("```java" in cpp_code):
                cpp_code = cpp_code.split("```java")[1]
            if ("```" in cpp_code):
                cpp_code = cpp_code.split("```")[0]
            if("Java Code:" in cpp_code):
                cpp_code = cpp_code.split("Java Code:")[1]
            cpp_code = re.sub('public\s*class\s*.+', 'public class ' + filename + ' {', cpp_code)
            with open(os.path.join(f"{script_file_path}/", f'{filename}.java'), 'w') as f:
                f.write(cpp_code)
        if (target == "cpp"):
            if ("```cpp" in cpp_code):
                cpp_code = cpp_code.split("```cpp")[1]
            if ("```" in cpp_code):
                cpp_code = cpp_code.split("```")[0]
            if ("C++ Code:" in cpp_code):
                cpp_code = cpp_code.split("C++ Code:")[1]

            with open(os.path.join(f"{script_file_path}/", f'{filename}.cpp'), 'w') as f:
                f.write(cpp_code)
        if (target == "python"):
            if ("```python" in cpp_code):
                cpp_code = cpp_code.split("```python")[1]
            if ("```" in cpp_code):
                cpp_code = cpp_code.split("```")[0]
            if ("Python Code:" in cpp_code):
                cpp_code = cpp_code.split("Python Code:")[1]
            if ("Python:" in cpp_code):
                cpp_code = cpp_code.split("Python:")[1]
            if ("python:" in cpp_code):
                cpp_code = cpp_code.split("python:")[1]
            if ("Python code:" in cpp_code):
                cpp_code = cpp_code.split("Python code:")[1]
            if ("python code:" in cpp_code):
                cpp_code = cpp_code.split("python code:")[1]
            if ("[PYTHON]" in cpp_code):
                cpp_code = cpp_code.split("[PYTHON]")[1]
            with open(os.path.join(f"{script_file_path}/", f'{filename}.py'), 'w') as f:
                f.write(cpp_code)

        print(f"Saved cpp code for '{id}' successfully.")
    except Exception as e:
        print(f"Error processing JSON: {e}")


# Read input file line by line
model = input("model")
# source = input("src")
# target = input("dst")

# shots = input("shots")
# stra = input("strategy:")

language_pairs = [
    # ("cpp", "python"),  # C++ to Python
    # ("python", "cpp"),  # Python to C++
    # ("java", "cpp"),  # Java to C++
    # ("cpp", "java"),  # C++ to Java
    ("python", "java"),  # Python to Java
    #("java", "python")  # Java to Python
]
# threshold = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
# for thre in threshold:
shots_list = [5]
selection_methods = ["random","codebert","coverage","bm25","similarity"]
# selection_methods = ["clustering"]

for source, target in language_pairs:
    for shots in shots_list:
        for stra in selection_methods:
            out_dir = f"/home/duyl/grammartran/avatar/cleaned_data/{model}/"
            input_file = out_dir + f"testable_{source}_{target}_w_syntax_fix_{shots}_shot_{stra}_selection.jsonl"
            if(os.path.exists(input_file)):
                # input_file = f'/home/duyl/grammartran/FSE-24-UniTrans-main/cleaned_data/{model}/testable_{source}_{target}_w_syntax_fix_oneshot.jsonl'
                with open(input_file, 'r') as f:
                    for line in f:
                        extract_and_save(line.strip(), model, source, target)