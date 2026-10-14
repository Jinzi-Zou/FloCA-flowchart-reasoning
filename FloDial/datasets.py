import json
import torch
from torch_geometric.data import InMemoryDataset, Data
import os.path as osp
import numpy as np
import copy
class TextGraphDataset(InMemoryDataset):
    def __init__(self, name: str, json_file: str, root="./cache_data", load_texts=True, transform=None, pre_transform=None):
        """
        初始化文本图数据集。

        :param name: 数据集名称
        :param json_file: 存储图结构的 JSON 文件路径
        :param root: 数据集存储根目录
        :param load_texts: 是否加载原始文本
        :param transform: 数据预处理变换函数
        :param pre_transform: 数据预处理前变换函数
        """
        self.graph_data = json_file
        self.name = name
        self.load_texts = load_texts
        self.root = root
        self.data_dir = osp.join(self.root, self.name)
        super().__init__(self.data_dir, transform, pre_transform)
        if osp.exists(self.processed_paths[0]):
            print(f"Loading processed data from {self.processed_paths[0]}")
            self.load()
        else:
            print("Processed data not found. Processing...")
            self.process()


    def process_data(self):
        """
        处理 JSON 数据并构建 PyG 图数据。

        :return: 图数据对象
        """

        # 处理节点数据
        description = self.graph_data["problem_description"]
        nodes = self.graph_data["nodes"]
        node_features = {}
        node_labels = {}
        node_terminals = {}

        for node_id, node_data in nodes.items():
            node_features[int(node_id)]=node_data["utterance"]
            node_labels[int(node_id)]=node_data["label"]
            node_terminals[int(node_id)] = node_data["terminal"]

        # 处理边数据
        edges = self.graph_data["edges"]
        edge_index = []
        edge_attr = []

        for node_id, neighbors in edges.items():
            for condition, neighbor_id in neighbors.items():
                edge_index.append([int(node_id), int(neighbor_id)])
                edge_attr.append([1 if condition == "yes" else 0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 创建图数据对象
        data = Data(
            x=None,  # 无需编码的节点特征
            edge_index=edge_index,  # 边索引
            edge_attr=edge_attr,  # 边属性
            node_labels=node_labels,  # 节点标签
            node_terminals=node_terminals,  # 是否终端节点
            raw_texts=node_features  # 直接存储原始文本
        )

        # 返回图数据和切片
        return data

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.name}_graph_data.pt"]

    def process(self):
        # 处理数据并将其赋值给 self.data 和 self.slices
        print("Processing data...")
        data = self.process_data()

        # 设置 self.data 和 self.slices
        self.data = data


        # 保存处理后的数据
        torch.save((self.data), self.processed_paths[0])
        print(f"Data saved at {self.processed_paths[0]}")

    def get(self, idx):
        # 返回图数据
        return self.data

    def load(self):
        self.data = torch.load(self.processed_paths[0], weights_only=False)


class Flowcharts(object):

    def __init__(self, flowchartJsons, glob=None):
        self._flowchartJsons = flowchartJsons
        if glob != None:
            self._max_node_utterance_length = glob['max_node_utterance_length']
            self._max_edge_utterance_length = glob['max_edge_label_length']
        else:
            self._max_node_utterance_length = -1
            self._max_edge_utterance_length = -1

        self._populate_node_properties()

    @property
    def max_node_utterance_length(self):
        return self._max_node_utterance_length

    @property
    def max_edge_utterance_length(self):
        return self._max_edge_utterance_length

    def _populate_node_properties(self):
        self._paths_to_root = {}
        self._node_to_text_map = {}
        for name, flowchart in self._flowchartJsons.items():
            parent_map = self._get_parents_map(flowchart)
            name = flowchart['name']
            self._paths_to_root[name] = {}
            self._node_to_text_map[name] = {}
            for node_id, node_properties in flowchart['nodes'].items():
                path_to_root = []
                curr_node = node_id
                while (curr_node in parent_map):
                    path_to_root.insert(0, parent_map[curr_node])
                    curr_node = parent_map[curr_node][0]
                self._paths_to_root[name][node_id] = copy.deepcopy(path_to_root)
                self._node_to_text_map[name][node_id] = node_properties['utterance']

    def _get_parents_map(self, flowchart):
        parent_map = {}
        for parent_node_id, edges in flowchart['edges'].items():
            for option, child_node_id in edges.items():
                parent_map[child_node_id] = (parent_node_id, option)
        return parent_map

    def get_node_text(self, flowchart, node_id):
        return self._node_to_text_map[flowchart][node_id]

    def get_all_node_ids(self, flowchart):
        return self._flowchartJsons[flowchart]['nodes'].keys()

    def get_edge_from_parent(self, flowchart, node_id):
        path_to_root = self._paths_to_root[flowchart][node_id]
        if len(path_to_root) == 0:
            return ""
        else:
            prev_node_edge_pair = path_to_root[-1]
            return prev_node_edge_pair[1]

    def get_parent_node_text(self, flowchart, node_id):
        path_to_root = self._paths_to_root[flowchart][node_id]
        if len(path_to_root) == 0:
            return ""
        else:
            prev_node_edge_pair = path_to_root[-1]
            return self._node_to_text_map[flowchart][prev_node_edge_pair[0]]

    def get_full_flowchart(self, flowchart):
        return self._flowchartJsons[flowchart]

    def get_path_to_node(self, flowchart, node_id):
        return self._paths_to_root[flowchart][node_id]

    def get_flowchart_names(self):
        return list(self._flowchartJsons.keys())

    def get_all_paths(self, flowchart):
        return self._paths_to_root[flowchart]

def process_flowcharts_for_paths(flowcharts, flowchartDocsJson):
    chart_data_dict = {}

    for chart in flowcharts.get_flowchart_names():
        chart_path_lengths = []
        chart_paths_as_text = []
        chart_responses_as_text = []
        for node in list(flowcharts.get_all_node_ids(chart)):
            path = flowcharts.get_path_to_node(chart, node)
            path_text = []
            path_lengths = []
            if len(path) == 0:
                text = flowcharts.get_node_text(chart,'1')
                path_text += [text]
            for n, e in path:

                path_text += [flowcharts.get_node_text(chart, n), e]
            chart_path_lengths.append(path_lengths)
            chart_paths_as_text.append(path_text)
            chart_responses_as_text.append(flowcharts.get_node_text(chart, node))

        # aDD DOCUMENTS
        for d in flowchartDocsJson[chart]:
            chart_path_lengths.append([path_lengths])
            chart_paths_as_text.append([d['q']])
            chart_responses_as_text.append(d['a'])

        chart_data_dict[chart] = {'paths_text': chart_paths_as_text,
                                  'responses_text': chart_responses_as_text}
    return chart_data_dict