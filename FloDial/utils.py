from os import listdir
from os.path import isfile, join, isdir
import torch
import numpy as np
import random
from datasets import *
from difflib import SequenceMatcher
from termcolor import colored


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
    flowchartsJson = read_flowchart_jsons(args.flowchart_dir)
    flowchartDocsJson = read_flowchart_doc_jsons(args.flowchart_dir)
    flowcharts = Flowcharts(flowchartsJson)
    chart_data_dic = process_flowcharts_for_paths(flowcharts,flowchartDocsJson)
    for chart_name,chart in flowchartsJson.items():
        dataset = TextGraphDataset(name=chart_name, json_file=chart)
        flowchartTool[chart_name]=dataset
        domain.append(chart_name)
    trnJson, valJson, tstJson = read_dialog_jsons(args.dialog_dir)
    return flowchartTool, flowchartDocsJson,trnJson,valJson,tstJson,domain,chart_data_dic,flowchartsJson


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

def find_first_diff(ground_node_path, pre_node_path):
    min_len = min(len(ground_node_path), len(pre_node_path))

    for i in range(min_len):
        if ground_node_path[i] != pre_node_path[i]:
            return pre_node_path[i],ground_node_path[i]

    # 如果前面都相同，但长度不同
    if len(ground_node_path) != len(pre_node_path):
        if len(ground_node_path) > min_len:
            return None, ground_node_path[min_len]
        else:
            return pre_node_path[min_len],None

    # 完全相同
    return None