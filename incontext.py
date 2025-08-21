from syntax_generator import get_type_list
import json
import random
import copy
import torch
from rank_bm25 import BM25Okapi
import numpy as np


from templates.examples_inp import example_cpp, valid_inputs_cpp, \
    example_java, valid_inputs_java, example_python, valid_inputs_python
from templates.examples_trans import \
    example_code_java, example_code_cpp, example_code_python, \
    example_test_cases_java, example_test_cases_cpp, example_test_cases_python
from templates.example_refine import python_refine_example2_1, python_refine_example2_2, \
    cpp_refine_example2_1, cpp_refine_example2_2, java_refine_example2_1, java_refine_example2_2
from syntax_generator import get_all_sub_trees, get_coverage_of_cand_tree, update_cur_refer_tree, ast_edit_distance, get_coverage_of_cand_dataflow, update_cur_refer_dataflow, get_data_flow, normalize_dataflow

from sklearn.metrics.pairwise import cosine_similarity

def fix_incontext_syntax(src_lang, dst_lang):
    example_src_code = eval(f"example_code_{src_lang}")
    example_dst_code = eval(f"example_code_{dst_lang}")
    example_src_syntax = get_type_list(example_src_code, src_lang)
    example_dst_syntax = get_type_list(example_dst_code, dst_lang)
    return example_src_code, example_src_syntax, example_dst_code, example_dst_syntax

def get_example_code(src_lang, dst_lang, data, codeimbeddings, item, outdir, selection, round, shots, threshold):
    # file_path = f"{outdir}{src_lang}_{dst_lang}_round{round-1}_results.jsonl"
    # data = {}
    # with open(file_path, encoding="utf-8") as fr:
    #     for line in fr.readlines():
    #         line = json.loads(line)
    #         data[line[src_lang]] = line[dst_lang]

    #data.pop(item, 'None')

    if(selection == "similarity"): #LD
        #similarity selection
        sim_src_codes = get_similar_incontext(item, data.keys(), shots)
        sim_dst_codes = [data[code] for code in sim_src_codes]
        return sim_src_codes, sim_dst_codes  # Return the destination codes for the top 'shots' most similar codes

    elif(selection == "codebert"):
        # similarity selection
        sim_src_codes = get_similar_incontext_by_codebert(item, data.keys(), shots, codeimbeddings)
        sim_dst_codes = [data[code] for code in sim_src_codes]
        return sim_src_codes, sim_dst_codes  # Return the destination codes for the top 'shots' most similar codes

    elif(selection == "random"):
        sim_src_codes = get_random_incontext(item, data.keys(), shots)
        sim_dst_codes = [data[code] for code in sim_src_codes]
        return sim_src_codes, sim_dst_codes  # Return the destination codes for the top 'shots' most similar codes

    elif(selection == "clustering"):
        # similarity selection
        src_codes = get_diverse_incontext_by_clustering(item, data.keys(), shots, codeimbeddings)
        dst_codes = [data[code] for code in src_codes]
        return src_codes, dst_codes  # Return the destination codes for the top 'shots' most similar codes

    elif (selection == "bm25"):
        # BM25 selection
        src_tokens = [code.split() for code in data.keys()]
        dst_tokens = [code.split() for code in data.values()]

        bm25 = BM25Okapi(src_tokens)

        query_tokens = item.split()

        scores = bm25.get_scores(query_tokens)

        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        top_indices = sorted_indices[:shots]

        bm25_src_codes = [list(data.keys())[idx] for idx in top_indices]
        bm25_dst_codes = [data[code] for code in bm25_src_codes]

        return bm25_src_codes, bm25_dst_codes  # Return the top 'shots' BM25-based similar codes

    elif(selection == "TED"): # AST ED
        #similarity selection
        sim_src_codes = get_similar_tree(item, data.keys(), src_lang, shots)
        sim_dst_codes = [data[code] for code in sim_src_codes]
        return sim_src_codes, sim_dst_codes  # Return the destination codes for the top 'shots' most similar codes


    elif(selection == "coverage"):
        #similarity selection
        sim_src_codes = get_similar_incontext_by_codebert(item, data.keys(), int(2 * shots),codeimbeddings)
        # coverage selection
        #cov_src_codes = get_coverage_incontext(item, list(data.keys()), src_lang, shots)
        cov_src_codes = get_coverage_incontext(item, sim_src_codes, src_lang, shots)
        cov_dst_codes = [data[code] for code in cov_src_codes]
        return cov_src_codes, cov_dst_codes  # Return the destination codes for the top 'shots' most similar codes

    elif (selection == "autocoverage"):
        # similarity selection
        sim_src_codes = get_similar_incontext(item, data.keys(), 2* shots)
        # coverage selection
        # cov_src_codes = get_coverage_incontext(item, list(data.keys()), src_lang, shots)
        cov_src_codes = auto_get_coverage_incontext(item, sim_src_codes, src_lang, shots, threshold)
        cov_dst_codes = [data[code] for code in cov_src_codes]
        return cov_src_codes, cov_dst_codes  # Return the destination codes for the top 'shots' most similar codes


    elif(selection == "onlyrandom"):
        #similarity selection
        src_codes = random.sample(data.keys(), shots)
        return coverage_score(item, src_codes, src_lang)
        
    elif(selection == "onlysimilarity"):
        #similarity selection
        sim_src_codes = get_similar_incontext(item, data.keys(), shots)
        return coverage_score(item, sim_src_codes, src_lang)

    elif (selection == "onlycodebert"):
        # similarity selection
        sim_src_codes = get_similar_incontext_by_codebert(item, data.keys(), shots, codeimbeddings)
        return coverage_score(item, sim_src_codes, src_lang)
        
    elif(selection == "onlycoverage"):
        #similarity selection
        sim_src_codes = get_similar_incontext_by_codebert(item, data.keys(), 2*shots, codeimbeddings)
        # coverage selection
        #cov_src_codes = get_coverage_incontext(item, list(data.keys()), src_lang, shots)
        cov_src_codes = get_coverage_incontext(item, sim_src_codes, src_lang, shots)
        return coverage_score(item, cov_src_codes, src_lang)

def similarity_score(itemcode, src_codes):
    sc = 0
    for code in src_codes:
        sc += levenshtein_distance(itemcode, code)
    return sc/len(src_codes)

def coverage_score(itemcode, src_codes, code_lang):
    # List to hold codes and their similarity scores
    origin_cur_refer_trees = get_all_sub_trees(itemcode, code_lang)
    cur_refer_trees = copy.deepcopy(origin_cur_refer_trees)
    candidiate_trees = {}
    for code in src_codes:
        candidiate_trees[code] = get_all_sub_trees(code, code_lang)
    for code in src_codes:
        cur_refer_trees = update_cur_refer_tree(candidiate_trees[code], cur_refer_trees)

    
    score1 = 1 - get_coverage_of_cand_tree(cur_refer_trees, origin_cur_refer_trees)
    # print(score1)
    return score1  


def auto_get_coverage_incontext(itemcode, input_src_codes, code_lang, maxshots, threshold):
    src_codes = copy.deepcopy(input_src_codes)
    top_codes = []
    # List to hold codes and their similarity scores
    origin_cur_refer_trees = get_all_sub_trees(itemcode, code_lang)
    cur_refer_trees = copy.deepcopy(origin_cur_refer_trees)
    candidiate_trees = {}
    for code in src_codes:
        candidiate_trees[code] = get_all_sub_trees(code, code_lang)

    for i in range(maxshots):
        best_score = -1.0
        best_code = src_codes[0]
        for code in src_codes:
            score1 = get_coverage_of_cand_tree(candidiate_trees[code], cur_refer_trees)
            score = score1
            if(score >= best_score):
                best_score = score
                best_code = code
        top_codes.append(best_code)
        cur_refer_trees = update_cur_refer_tree(candidiate_trees[best_code], cur_refer_trees)
        coverage_score = 1 - get_coverage_of_cand_tree(cur_refer_trees, origin_cur_refer_trees)
        if(coverage_score >= threshold):
            return top_codes
        src_codes.remove(best_code)

        #cur_refer_dfg = update_cur_refer_dataflow(candidiate_dfgs[best_code], cur_refer_dfg)
    return top_codes  # Return the top 'shots' most coverage source codes



def get_coverage_incontext(itemcode, input_src_codes, code_lang, shots):
    src_codes = copy.deepcopy(input_src_codes)
    top_codes = []
    # List to hold codes and their similarity scores
    cur_refer_trees = get_all_sub_trees(itemcode, code_lang)
    candidiate_trees = {}
    for code in src_codes:
        candidiate_trees[code] = get_all_sub_trees(code, code_lang)

    for i in range(shots):
        best_score = -1.0
        best_code = src_codes[0]
        for code in src_codes:
            score1 = get_coverage_of_cand_tree(candidiate_trees[code], cur_refer_trees)
            #score2 = get_coverage_of_cand_dataflow(candidiate_dfgs[code], cur_refer_dfg)
            #score3 = levenshtein_distance(itemcode, code)/(len(code)+len(itemcode))
            score = score1
            #print(score, "syntaxtree", score1, "levenshtein", score3)

            if(score >= best_score):
                best_score = score
                best_code = code
        top_codes.append(best_code)
        cur_refer_trees = update_cur_refer_tree(candidiate_trees[best_code], cur_refer_trees)
        src_codes.remove(best_code)

        #cur_refer_dfg = update_cur_refer_dataflow(candidiate_dfgs[best_code], cur_refer_dfg)
    return top_codes  # Return the top 'shots' most coverage source codes


def get_similar_incontext(itemcode, src_codes, shots):
    # List to hold codes and their similarity scores
    code_scores = []

    # Calculate similarity for each source code
    for code in src_codes:
        score = levenshtein_distance(itemcode, code)  # Or another similarity function
        code_scores.append((code, score))

    # Sort by score (ascending), then extract the top 'shots' codes
    code_scores.sort(key=lambda x: x[1])
    top_codes = [code for code, score in code_scores[:shots]]

    return top_codes  # Return the top 'shots' most similar source codes

def get_similar_incontext_by_codebert(itemcode, src_codes, shots, codeimbeddings):
    # List to hold codes and their similarity scores
    code_scores = []

    # Calculate similarity for each source code
    for code in src_codes:
        try:
            score = codebert_distance(itemcode, code, codeimbeddings)  # Or another similarity function
            code_scores.append((code, score))
        except:
            continue

    # Sort by score (ascending), then extract the top 'shots' codes
    code_scores.sort(key=lambda x: x[1])
    top_codes = [code for code, score in code_scores[:shots]]

    return top_codes  # Return the top 'shots' most similar source codes

def get_similar_tree(itemcode, src_codes, code_lang, shots):
   # List to hold codes and their similarity scores
    code_scores = []

    # Calculate similarity for each source code using AST edit distance
    for code in src_codes:
        score = ast_edit_distance(itemcode, code, code_lang)
        code_scores.append((code, score))
    # Sort by score (ascending), then extract the top 'shots' codes
    code_scores.sort(key=lambda x: x[1])
    top_codes = [code for code, score in code_scores[:shots]]

    return top_codes  # Return the top 'shots' most similar source codes


from sklearn.cluster import KMeans
import numpy as np

def get_diverse_incontext_by_clustering(itemcode, src_codes, shots, codeembbedings):
    """
    Get diverse in-context examples based on clustering of source codes.

    Parameters:
    - itemcode: The target item for which we want to find similar examples.
    - src_codes: List or array of source codes to be clustered.
    - shots: The number of similar examples to return.
    - k: The number of clusters for KMeans.

    Returns:
    - A list of 'shots' number of most similar samples to itemcode from the clustered src_codes.
    """
    
    # Assuming src_codes is a list of vectors or embeddings
    # Step 1: Perform KMeans clustering on the src_codes
    kmeans = KMeans(n_clusters=shots, random_state=42)
    dbembedding = []
    for code1 in src_codes:
        try:
            dbembedding.append(codeembbedings[code1])
        except:
            continue
    dbembedding = torch.stack(dbembedding, dim=0).squeeze()
    # print(dbembedding.shape)
    kmeans.fit(dbembedding)
    
        # Step 2: Find the cluster that contains the itemcode (find its closest cluster center)
    item_embedding = codeembbedings[itemcode]
    
    # Predict the cluster of the itemcode
    itemcode_cluster = kmeans.predict(item_embedding.reshape(1, -1))[0]
    
    # Step 3: Select the most similar sample from each cluster
    selected_samples = []
    
    for cluster_id in range(shots):
        # Get the indices of the samples in this cluster
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        
        # Compute cosine similarity between itemcode's embedding and all samples in the cluster
        cluster_embeddings = dbembedding[cluster_indices]
        similarities = cosine_similarity(item_embedding.reshape(1, -1), cluster_embeddings)
        
        # Find the most similar sample (highest cosine similarity)
        most_similar_idx = cluster_indices[np.argmax(similarities)]
        
        # Add the corresponding sample to the selected list
        selected_samples.append(list(src_codes)[most_similar_idx])
    
    return selected_samples



def codebert_distance(code1, code2, codeimbeddings):
    # 获取两个代码片段的嵌入
    embedding1 = codeimbeddings[code1]
    embedding2 = codeimbeddings[code2]

    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]


def levenshtein_distance(s1, s2):
    # Example Levenshtein distance implementation (can replace with a better method)
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]



def get_random_incontext(itemcode, src_codes, shots):
    # List to hold codes and their similarity scores
    return random.sample(src_codes,shots)

