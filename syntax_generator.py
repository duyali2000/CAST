

from __future__ import absolute_import
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import pickle
import torch
import copy
from re import sub
from nltk.stem.porter import *
from nltk.corpus import stopwords
import json
import csv
import random
import logging
import argparse
# import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)
from tree_sitter import Language, Parser
from typing import List, Optional
sys.path.append('..')


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


from parser import DFG_python,DFG_java,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser


dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'cpp': DFG_csharp
}

logger = logging.getLogger(__name__)
# load parsers
funcs = ['python', 'java', 'cpp']
parsers = {}
for lang in funcs:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    # parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        # code = code.split('\n')
        # code_tokens , type_tokens = [], []
        # for x in tokens_index:
        #     code_, type_ = index_to_code_token(x, code)
        #     code_tokens.append(code_)
        #     type_tokens.append(type_)

    except:
        print("syntax error")
    #return code_tokens, type_tokens
    return tokens_index


def filter_syntax(source_code, code_lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(source_code, code_lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parsers[code_lang].parse(bytes(code, 'utf8'))
    except:
        return False
    else:
        return True


def get_type_list(source_code, code_lang):
    #code_tokens
    type_tokens = extract_dataflow(source_code,
                                                parsers[code_lang],
                                                code_lang)
    # print(code_tokens, type_tokens)
    return type_tokens


def get_all_sub_trees(candidate, code_lang):
    try:
        candidate = remove_comments_and_docstrings(candidate, code_lang)
    except Exception:
        pass
    candidate_tree = parser.parse(bytes(candidate, "utf8")).root_node
    def get_all_sub_trees(root_node):
        node_stack = []
        sub_tree_sexp_list = []
        depth = 1
        node_stack.append([root_node, depth])
        while len(node_stack) != 0:
            cur_node, cur_depth = node_stack.pop()
            sub_tree_sexp_list.append([str(cur_node.type), cur_depth])
            for child_node in cur_node.children:
                if len(child_node.children) != 0:
                    depth = cur_depth + 1
                    node_stack.append([child_node, depth])
        return sub_tree_sexp_list

    cand_sexps = [x[0] for x in get_all_sub_trees(candidate_tree)]
    return cand_sexps

def get_coverage_of_cand_tree(cand_sexps, ref_sexps):
    # 使用集合的交集来计算重复元素
    match_count = len(set(ref_sexps) & set(cand_sexps))  # 交集操作
    total_count = len(ref_sexps)

    if total_count == 0:
        return 0
    score = match_count / total_count
    return score


def update_cur_refer_tree(minus_sub_trees, ref_sexps):
    # 使用集合差集来移除元素
    out_ref_sexps = list(set(ref_sexps) - set(minus_sub_trees))  # 差集操作
    return out_ref_sexps



def get_coverage_of_cand_dataflow(cand_dfg, ref_dfg):
    match_count, total_count = 0, 0
    if len(ref_dfg) > 0:
        total_count += len(ref_dfg)
        for dataflow in ref_dfg:
            if dataflow in cand_dfg:
                match_count += 1
                cand_dfg.remove(dataflow)
    if (total_count == 0):
        return 0
    score = match_count / total_count
    return score

def update_cur_refer_dataflow(minus_sub_dfg, ref_dfg):
    out_ref_dfg = ref_dfg
    for dataflow in minus_sub_dfg:
        if(dataflow in ref_dfg):
            out_ref_dfg.remove(dataflow)
    return out_ref_dfg


def get_data_flow(code, code_lang):
    parser = parsers[code_lang]
    parser = [parser, dfg_function[code_lang]]

    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except Exception:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except Exception:
        #code.split()
        dfg = []
    # merge nodes
    dic = {}
    for d in dfg:
        if d[1] not in dic:
            dic[d[1]] = d
        else:
            dic[d[1]] = (
                d[0],
                d[1],
                d[2],
                list(set(dic[d[1]][3] + d[3])),
                list(set(dic[d[1]][4] + d[4])),
            )
    DFG = []
    for d in dic:
        DFG.append(dic[d])
    dfg = DFG
    return dfg

def normalize_dataflow_item(dataflow_item):
    var_name = dataflow_item[0]
    dataflow_item[1]
    relationship = dataflow_item[2]
    par_vars_name_list = dataflow_item[3]
    dataflow_item[4]

    var_names = list(set(par_vars_name_list + [var_name]))
    norm_names = {}
    for i in range(len(var_names)):
        norm_names[var_names[i]] = "var_" + str(i)

    norm_var_name = norm_names[var_name]
    relationship = dataflow_item[2]
    norm_par_vars_name_list = [norm_names[x] for x in par_vars_name_list]

    return (norm_var_name, relationship, norm_par_vars_name_list)

def normalize_dataflow(dataflow):
    var_dict = {}
    i = 0
    normalized_dataflow = []
    for item in dataflow:
        var_name = item[0]
        relationship = item[2]
        par_vars_name_list = item[3]
        for name in par_vars_name_list:
            if name not in var_dict:
                var_dict[name] = "var_" + str(i)
                i += 1
        if var_name not in var_dict:
            var_dict[var_name] = "var_" + str(i)
            i += 1
        normalized_dataflow.append(
            (
                var_dict[var_name],
                relationship,
                [var_dict[x] for x in par_vars_name_list],
            )
        )
    return normalized_dataflow




def _ast_node_distance(node1, node2) -> int:
    """计算两个tree-sitter节点之间的距离（支持多语言）"""
    if node1 is None and node2 is None:
        return 0
    if node1 is None or node2 is None:
        return 1
    
    # 节点类型不同，基础代价（多语言通用）
    cost = 0 if node1.type == node2.type else 1
    
    # 比较节点文本内容（标识符、关键字、字面量等）
    # 不同语言的文本含义可能不同，但统一比较文本值
    if node1.text != node2.text:
        cost += 1
    
    return cost

def _ast_node_type_distance(type1: str, type2: str) -> int:
    """计算两个节点类型的距离（仅比较类型字符串）"""
    return 0 if type1 == type2 else 1  # 类型相同代价0，不同代价1

def _sequence_ast_distance(seq1_types: List[str], seq2_types: List[str]) -> int:
    """计算两个“节点类型序列”的编辑距离（动态规划核心）"""
    m, n = len(seq1_types), len(seq2_types)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界：空序列与非空序列的距离=非空序列长度（全插入/删除）
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # 填充DP表：仅基于节点类型计算代价
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            type_cost = _ast_node_type_distance(seq1_types[i-1], seq2_types[j-1])
            dp[i][j] = min(
                dp[i-1][j] + 1,          # 删除：移除seq1的当前类型
                dp[i][j-1] + 1,          # 插入：给seq1插入seq2的当前类型
                dp[i-1][j-1] + type_cost # 替换：类型相同代价0，不同代价1
            )
    
    return dp[m][n]

def _parse_code_to_ast(code: str, language: str):
    """将代码字符串解析为指定语言的AST根节点"""
    parser = parsers[language]
    if not parser:
        return None
    try:
        tree = parser.parse(bytes(code, "utf8"))
        return tree.root_node
    except Exception:
        return None

def ast_edit_distance(code1: str, code2: str, language: str) -> int:
    """计算两种代码的AST编辑距离（支持多语言）"""
    # 解析为指定语言的AST
    ast1 = _parse_code_to_ast(code1, language)
    ast2 = _parse_code_to_ast(code2, language)
    
    # 处理解析失败的情况
    if ast1 is None and ast2 is None:
        return 0
    if ast1 is None or ast2 is None:
        return float('inf')  # 一方解析失败，视为差异极大
    
     # 关键：提取根节点子节点的“类型序列”（而非节点本身）
    ast1_child_types = [child.type for child in ast1.children]
    ast2_child_types = [child.type for child in ast2.children]
    
    # 计算类型序列的编辑距离
    return _sequence_ast_distance(ast1_child_types, ast2_child_types)


if __name__ == "__main__":
    source_code = """class Javahelloworld {
    public static void main(String args[]){
    System.out.println("hello world\n");
    }
    }"""
    # code_tokens, type_tokens = extract_dataflow(source_code,
    #                                         parsers["java"],
    #                                         "java")
    # # print(code_tokens, type_tokens)
    # for code_, type_ in enumerate(zip(code_tokens, type_tokens)):
    #     print(f"<{code_}:{type_}>")

    type_list = get_type_list(source_code, "java")
    print(type_list)


