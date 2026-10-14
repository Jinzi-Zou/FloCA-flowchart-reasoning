import re
from utils import *
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from agent_tool import agent_toolselction
from agent_node_location import agent_node_location
from user import User
from agent_chat import agent_chat
from agent_qa import agent_qa
from collections import defaultdict
from itertools import islice

from difflib import SequenceMatcher


def parge_args():
    parser = argparse.ArgumentParser(description='Train and evaluate FlowNet')
    parser.add_argument('--flowchart-dir', type=str, default='./FloDial-dataset/knowledge-sources/', help='a directiory that contains all flowcharts')
    parser.add_argument('--dialog-dir', type=str, default='./FloDial-dataset/dialogs/',
                        help='a directiory that contains trn, val and tst dialogs')
    parser.add_argument('--model-name', type=str, required=True,
                        help='name of the model used for inference')
    parser.add_argument('--openrouter-base-url', type=str, default='',
                        help='OpenAI-compatible API base URL')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='dialog index from which evaluation resumes')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_all_seeds()
    sys.stdout.flush()
    args = parge_args()

    flowchartTool, flowchartDocsJson,trnJson,valJson,tstJson,domain,chart_data_dic,flowchartsJson = get_flowchart_tool(args)
    Model_NAME = args.model_name
    OPENROUTER_BASE_URL = args.openrouter_base_url
    api_key = os.getenv('MODEL_API_KEY', '')

    remote_models = {
        'openai/gpt-4o',
        'openai/gpt-5',
        'anthropic/claude-opus-4.1',
        'deepseek-reasoner',
        'qwen-plus',
    }
    if Model_NAME in remote_models and not api_key:
        raise ValueError('MODEL_API_KEY environment variable is required for remote models')

    tool_disc = {}
    for inx, (name, tool) in enumerate(flowchartTool.items()):
        tool_disc[name]=(f"{tool.graph_data['problem_description']}")
    agent_toolsl = agent_toolselction(tool_disc,domain,Model_NAME,OPENROUTER_BASE_URL,api_key)
    pre_tool_name = []
    ground_tool_name = []
    user_first_input = []
    #打乱测试对话顺序：
    keys = list(tstJson.keys())
    # num_sample = 3
    random.shuffle(keys)
    shuffled_tstJson = {k: tstJson[k] for k in keys}
    bleu_scores = []
    # doc_label = []
    batch_index_pre = []
    pred_node = []
    agent_response_l = {}
    first_doct_label = {}
    error_tool = []
    print(f'共{len(shuffled_tstJson)}个对话样本')
    pre_dialog = defaultdict(dict)
    checkpoint = args.checkpoint

    safe_model_name = Model_NAME.replace('/', '_')
    results_dir = Path('./results') / safe_model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'pre_test_dialogs_{safe_model_name}_{checkpoint}.json'

    for index, (dialogue_id, dialogue) in tqdm(enumerate(islice(shuffled_tstJson.items(),checkpoint, None),start=checkpoint)):
        doc_label = []
        print(f'\033[91m dialog:{index} \033[0m')
        print(f'\033[91m dialog原始id:{dialogue_id} \033[0m')

        current_node_id = None  # 每个对话开始时，当前节点应该初始化为 None

        user_utterances = {}  # 用于存储每轮用户输入
        agent_utterances = {}  # 用于存储每轮代理回复
        node_length = len(flowchartTool[dialogue['flowchart']].graph_data["nodes"])
        # 收集每轮对话中的用户和代理的utterances
        for turn in dialogue['utterences']:
            turn_id = turn['turn_id']
            if turn['speaker'] == 'user':
                user_utterances[turn_id] = turn['utterance']

            elif turn['speaker'] == 'agent':
                agent_utterances[turn_id] = turn['utterance']
                if "grounded_doc_id" in turn:
                    id = int(re.search(r'\d+', turn['grounded_doc_id']).group())
                    if turn['grounded_doc_id'].startswith('faq'):
                        id = id+node_length
                    doc_label.append(id)
                else:
                    doc_label.append('End')

        dialogu_turns = len(user_utterances)
        turn_budget = 2*dialogu_turns

        tool_name = dialogue['flowchart']
        toolgraph = flowchartTool[tool_name][0]
        QA_tool = flowchartDocsJson[tool_name]

        chart_data = chart_data_dic[tool_name]
        chart_dict = flowchartsJson[tool_name]


        agent_node_location_result = agent_node_location(toolgraph,Model_NAME,OPENROUTER_BASE_URL,api_key)
        user = User(dialogue,chart_data,chart_dict)

        #结果记录
        turn_record = []
        last_node_record = []
        first_node_recrd = []
        turn_count=0
        utterences = []
        while True:
            turn_pre_user = {}
            turn_pre_agent = {}
            if turn_count == 0:
                user_input = user_utterances[1]
                #定位第一个节点
                new_node_id = agent_node_location_result.first_node_loc(user_input)
                reference_response = toolgraph.raw_texts[new_node_id]  #获取第一个节点对应的文本内容
                chat_bot=agent_chat(Model_NAME,OPENROUTER_BASE_URL,api_key)
                chat_agent = chat_bot.prepare_agent_chat()  #初始化chatbot
                agent_response = chat_bot.chat_predict(reference_response,user_input) #用户输入，生成响应
                print(agent_response)
                # AI agent输出文本清洗
                agent_response = re.sub(r"<think>.*?</think>", "", agent_response, flags=re.DOTALL).strip()
                agent_response = agent_response.split('\n')[0]
                print(f"============turn{turn_count + 1}===============")
                print(f"\033[31mUser\033[0m:{user_input}")
                print(f"\033[34mAI assistance\033[0m:{agent_response}")
                turn_count+=1
                current_node_id = new_node_id
                first_node_id = new_node_id
                new_question = agent_response
                turn_pre_user["speaker"]="user"
                turn_pre_user["utterance"]=user_input
                turn_pre_user["turn_id"] = turn_count
                turn_pre_user["pre_doc_id"] = "chart-0"
                turn_pre_agent["speaker"]="agent"
                turn_pre_agent["utterance"] = agent_response
                turn_pre_agent["turn_id"] = turn_count
                turn_pre_agent["pre_doc_id"] = f"chart-{current_node_id}"
                utterences.extend([turn_pre_user,turn_pre_agent])
            else:
                if toolgraph.node_terminals[current_node_id] is True:
                    break
                if new_node_id!=-1:
                    unanswerd_question='None'
                user_input = user.user_simulator.predict(input=agent_response, unanswerd_question = unanswerd_question, stop=['AI Assistant', 'user over'])
                is_troubleshooting_successful = 'Troubleshooting successful' in user_input
                # 用户模拟器输出文本清洗
                user_input = user_input.split('\n')[0]
                user_input = re.sub(r"\*\*User's question\*\*", '', user_input)
                new_node_id = agent_node_location_result.update_node_loc(user_input,new_question,current_node_id,QA_tool)
                if new_node_id == -1:
                    unanswerd_question = new_question
                    qa_bot = agent_qa(Model_NAME,OPENROUTER_BASE_URL,api_key).prepare_agent_qa(QA_tool)
                    reference_response = qa_bot.predict(agent_response=new_question,input=user_input)
                    best_match = max(QA_tool, key=lambda x: SequenceMatcher(None, x['a'], reference_response).ratio())
                    turn_pre_agent["pre_doc_id"] = f"faq-{best_match['id']}"
                else:
                    reference_response = toolgraph.raw_texts[new_node_id]  # 获取跳转节点对应的文本内容
                    turn_pre_agent["pre_doc_id"] = f"chart-{new_node_id}"

                agent_response = chat_bot.chat_predict(reference_response, user_input)
                # AI agent输出文本清洗
                agent_response = re.sub(r"<think>.*?</think>", "", agent_response, flags=re.DOTALL).strip()
                agent_response = agent_response.split('\n')[0]
                if new_node_id != -1:
                    new_question = agent_response
                    current_node_id = new_node_id
                print(f"============turn{turn_count + 1}===============")
                print(f"\033[31mUser\033[0m:{user_input}")
                print(f"\033[34mAI assistance\033[0m:{agent_response}")
                turn_count += 1
                turn_pre_user["speaker"] = "user"
                turn_pre_user["utterance"] = user_input
                turn_pre_user["turn_id"] = turn_count
                turn_pre_user["pre_doc_id"] = "chart-0"
                turn_pre_agent["speaker"] = "agent"
                turn_pre_agent["utterance"] = agent_response
                turn_pre_agent["turn_id"] = turn_count
                utterences.extend([turn_pre_user, turn_pre_agent])

                if turn_count>turn_budget:
                    break
        pre_dialog[dialogue_id]["utterences"]=utterences
        pre_dialog[dialogue_id]["flowchart"]=tool_name
        print("对话结束")

        save_dict_to_json(pre_dialog, str(results_file))
