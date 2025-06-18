from bert_score import score


import pickle
import sys
sys.path.append('..')

# f = open('../gpt_test.pkl', 'rb')
# gpt_dict = pickle.load(f)
#
# fl = open('../../data/test.pkl', 'rb')
# examples = pickle.load(fl)
#
# predicts, targets = [], []
# for (code, summary) in examples:
#     if(code in gpt_dict.keys()):
#         presummary = gpt_dict[code].strip()
#         if(len(presummary) < 2):
#             break
#         presummary = presummary.replace("commit message: ","")
#         print(presummary)
#         print(summary)
#         predicts.append(presummary)
#         targets.append(summary)

import os
path = "/home/duyl/CommitMsg/OurModel/saved_models_codebert"

with open(os.path.join(path, "dev.output"), 'r') as f, open(os.path.join(path, "dev.gold"), 'r') as f1:
    predicts = f.readlines()
    targets = f1.readlines()

P, R, F1 = score(predicts, targets,model_type="bert-base-chinese",lang="zh", verbose=True)
print(F1)
print(f"System level F1 score: {F1.mean():.3f}")
# cands 和refs一一对应
# tensor([0.9148, 1.0000])
# System level F1 score: 0.957
