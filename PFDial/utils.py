from os import listdir
from os.path import isfile, join, isdir
import torch
import numpy as np
import random
from datasets import *
from difflib import SequenceMatcher
from termcolor import colored
import re

def read_dialog_jsons(dialog_dir):
    tstJson_list = read_json(dialog_dir+'s-flo-test.json')['tst']
    trn_val_Json = read_json(dialog_dir+'s-flo.json')
    dialog_whole=read_json(dialog_dir+'dialogs.json')
    dialog_2=read_json(dialog_dir+'test_dialogs.json')
    dialog_whole.update(dialog_2)
    trnJson = {k: dialog_whole[k] for k in trn_val_Json['trn'] if k in dialog_whole}
    valJson = {k: dialog_whole[k] for k in trn_val_Json['val'] if k in dialog_whole}
    tstJson = {k: dialog_whole[k] for k in tstJson_list if k in dialog_whole}
    return trnJson, valJson, tstJson

def read_flowchart_jsons(flowchart_dir):
    # load flowcharts from file
    flowcharts = {}
    flowchart_files = get_json_files(flowchart_dir)
    for flowchart_file in flowchart_files:
        flowchart_json = read_json(flowchart_file)
        flowcharts[flowchart_json['name']] = flowchart_json
    return flowcharts

def get_json_files(folder):
   return [folder+f for f in listdir(folder) if (isfile(join(folder, f)) and f.endswith('.json'))]

def read_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
        return data
def save_dict_to_json(data_dict, filename,indent=4,ensure_ascii=False):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=indent, ensure_ascii=ensure_ascii)
    return
def save_set_to_file(data_set, filename):
    """将集合保存到txt文件，每个元素一行"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_set:
            f.write(f"{item}\n")
def parse_response(text: str):
    pattern = r"下一状态：\s*(.*?)\s*\n\s*提示：\s*(.*)"
    match = re.search(pattern, text, re.S)

    if not match:
        raise ValueError(f"无法解析模型输出：\n{text}")

    next_state = match.group(1).strip()
    hint = match.group(2).strip()

    return next_state, hint

def read_flowchart_doc_jsons(flowchart_dir):
    # load flowcharts from file
    flowchart_docs = {}
    flowchart_files = get_json_files(flowchart_dir)
    for flowchart_file in flowchart_files:
        flowchart_json = read_json(flowchart_file)
        flowchart_docs[flowchart_json['name']]=flowchart_json["supporting_faqs"]
    return flowchart_docs

def set_all_seeds(seed=1):
    """固定所有随机种子以确保实验可复现"""
    # 1. Python内置随机种子
    random.seed(seed)

    # 2. NumPy随机种子
    np.random.seed(seed)

    # 3. PyTorch随机种子（CPU和CUDA）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时使用

def get_flowchart_tool(args):
    flowchartTool = {}
    domain = []
    flowchartsJson = read_json(args.flowchart_dir)
    for chart_name,chart in flowchartsJson.items():
        dataset = TextGraphDataset(name=chart_name, json_file=chart)
        flowchartTool[chart_name]=dataset
        domain.append(chart_name)
    tstJson = read_json(args.dialog_dir)
    return flowchartTool,tstJson,domain,flowchartsJson


def most_similar_string(clean_response, domain):
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    similarities = [similarity(clean_response, d) for d in domain]
    best_index = similarities.index(max(similarities))
    return domain[best_index]

def tenacity_retry_log(retry_state):
    t = retry_state.next_action.sleep
    e = retry_state.outcome.exception()
    msg = f'Tenacity: Retrying call Agent in {t:.2f} seconds as it raise {e.__class__.__name__}: '
    msg = colored(msg, 'red', force_color=True) + str(e)
    print(msg)

def find_next_node(data,current_node_id, edge_value):
    #data, next_node_id, edge_value=1

    edge_index = data.edge_index  # Shape [2, num_edges]
    edge_attr = data.edge_attr  # Shape [num_edges]

    # Find the edges connected to the current node (either as a source or destination)
    # The first row of edge_index contains the source nodes, and the second row contains the destination nodes.
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Find all the edges where the current node is either the source or the target node
    # relevant_edges = (source_nodes == current_node_id) | (target_nodes == current_node_id)
    relevant_edges = (source_nodes == current_node_id)
    # Filter edges by the specified edge_value (0 or 1)
    edge_attr = edge_attr.view(-1)
    valid_edges = relevant_edges & (edge_attr == edge_value)

    # Get the target node ids for the valid edges
    next_nodes = target_nodes[valid_edges]

    if next_nodes.size(0) == 0:
        return None  # No valid next node found
    else:
        return next_nodes.item()  # Return the ID of the next node

def cal_accuracy(pre,truth):
    correct = 0
    for p, t in zip(pre, truth):
        if p == t:
            correct += 1


    accuracy = correct / len(truth)
    return accuracy

def normalize_text(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "")
        .replace("：", ":")
        .replace("；", "")
        .replace("\n", "")
    )

def match_state_to_node_id(
    state_text: str,
    raw_texts: dict,
    similarity_threshold: float = 0.6
):
    """
    state_text: 模型输出的状态名称
    raw_texts: toolgraph.raw_texts {node_id(str): node_text}
    """

    norm_state = normalize_text(state_text)

    # ---------- 1. 完全匹配 ----------
    for node_id, node_text in raw_texts.items():
        if norm_state == normalize_text(node_text):
            return node_id

    # ---------- 2. 子串匹配 ----------
    for node_id, node_text in raw_texts.items():
        norm_node = normalize_text(node_text)
        if norm_state in norm_node or norm_node in norm_state:
            return node_id

    # ---------- 3. 相似度匹配 ----------
    best_id = None
    best_score = 0.0

    for node_id, node_text in raw_texts.items():
        score = SequenceMatcher(
            None, norm_state, normalize_text(node_text)
        ).ratio()
        if score > best_score:
            best_score = score
            best_id = node_id

    if best_score >= similarity_threshold:
        return best_id

    # ---------- 4. 匹配失败 ----------
    return None