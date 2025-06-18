The replication package includes code, dataset, etc. of CAST. In this work, we focus on post-incorporating code structural knowledge into pre-trained models via ICL for code translation. We revisit exemplar selection in ICL from an information-theoretic perspective, showing that maximizing information coverage provides a more precise and general solution than traditional methods. We introduce CAST, a surrogate measure based on the maximum subtree coverage ratio of AST. We show that NP-hard CAST maximization is a submodular maximization problem and propose a greedy algorithm with a (1 − 1/e)-approximate solution of polynomial time complexity. Our method is the first training-free, model-agnostic approach to integrate code structural knowledge into existing LLMs at test time, and both theory and experiments prove their effectiveness and efficiency. This work provides two key insights: 1) Code structural knowledge can be post-incorporated into pre-trained LLMs during inference despite training neglect. 2) Scaling model size or data doesn’t lead to code structural knowledge, highlighting the need to consider code syntax in LLMs.

        
## Running the Artifact

* Use different scripts to perform LLM-powered code translation, selection, compilation, and feedback evaluation. The instructions of commands have been clearly stated in the code. Please modify paths and parameters as needed.

### 1. Generate Code with LLM (Few-shot Prompting)
```bash
python method_few_shot.py --src_lang cpp --dst_lang python --obj 3 --k 10 --model_name qwen2:7b --shots 5 --round 2 --selection [ICL strategy] --shortname qwen7b
```

### 2. Extract Generated Code
```bash
python selectfile.py
```

### 3. Compile Code and Collect Success Rate
```bash
python compile_avatar.py --source_lang java --target_lang python --model qwen7b --shots 5 --selection random --report_dir ./report
```

### 4. Get Compilation Error Feedback
```bash
python compile_avatar_feedback.py --source_lang java --target_lang python --model qwen2.5 --shots 0 --selection none --report_dir ./report --attempt 1
python compile_codenet_feedback.py --source_lang java --target_lang python --model llama3.1 --shots 0 --selection COT --report_dir ./report --attempt 1
```
