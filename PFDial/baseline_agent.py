from utils import *
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from agent_node_location import agent_node_location
from user import User
from agent_chat import agent_chat
from collections import defaultdict
from itertools import islice

def parge_args():
    parser = argparse.ArgumentParser(description='Train and evaluate FlowNet')
    parser.add_argument('--flowchart-dir', type=str, default='./PFDial_datasets/flowchart/flowchart_test_ood_formatA(main).json', help='a directiory that contains all flowcharts')
    parser.add_argument('--dialog-dir', type=str, default='./PFDial_datasets/dialog/annotated_dialog_test_ood_formatA(main).json',
                        help='a directiory that contains trn, val and tst dialogs')
    parser.add_argument('--model-name', type=str, required=True,
                        help='name of the model used for inference')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='dialog index from which evaluation resumes')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    set_all_seeds()
    sys.stdout.flush()
    args = parge_args()

    flowchartTool,tstJson,domain,flowchartsJson = get_flowchart_tool(args)
    Model_NAME = args.model_name
    api_key = os.getenv('MODEL_API_KEY', '')

    if Model_NAME in {'openai/gpt-4o','openai/gpt-5','anthropic/claude-opus-4.1'}:
        OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    elif Model_NAME=='deepseek-reasoner':
        OPENROUTER_BASE_URL=''
    elif Model_NAME=='qwen-plus':
        OPENROUTER_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    else:
        OPENROUTER_BASE_URL=''

    remote_models = {
        'openai/gpt-4o',
        'openai/gpt-5',
        'anthropic/claude-opus-4.1',
        'deepseek-reasoner',
        'qwen-plus',
    }
    if Model_NAME in remote_models and not api_key:
        raise ValueError('MODEL_API_KEY environment variable is required for remote models')

    pre_tool_name = []
    ground_tool_name = []
    user_first_input = []
    #打乱测试对话顺序：
    keys = list(tstJson.keys())
    random.shuffle(keys)
    shuffled_tstJson = {k: tstJson[k] for k in keys}
    bleu_scores = []
    batch_index_doc = []
    batch_index_pre = []
    pred_node = []
    agent_response_l = {}
    first_doct_label = {}
    error_tool = []
    print(f'共{len(shuffled_tstJson)}个对话样本')
    pre_dialog = defaultdict(dict)
    checkpoint = args.checkpoint

    safe_model_name = Model_NAME.replace('/', '_').replace(':', '_')
    results_dir = Path('./results') / safe_model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'pfdial_pre_test_dialogs_{safe_model_name}_{checkpoint}.json'

    for index, (dialogue_id, dialogue) in tqdm(
            enumerate(islice(shuffled_tstJson.items(), checkpoint, None), start=checkpoint)):
        doc_label = []
        print(f'\033[91m dialog:{index} \033[0m')
        print(f'\033[91m dialog原始id:{dialogue_id} \033[0m')
        current_node_id = None

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


        dialogue_turn= len(user_utterances)
        turn_budget = 2*dialogue_turn

        tool_name = dialogue['flowchart']
        toolgraph = flowchartTool[tool_name][0]
        chart_dict = flowchartsJson[tool_name]

        agent_node_location_result = agent_node_location(toolgraph,Model_NAME,OPENROUTER_BASE_URL,api_key)
        user = User(dialogue)

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
                reference_response = toolgraph.raw_texts[str(new_node_id)]  #获取第一个节点对应的文本内容
                chat_bot=agent_chat(toolgraph,Model_NAME,OPENROUTER_BASE_URL,api_key)
                chat_agent = chat_bot.prepare_agent_chat()  #初始化chatbot
                agent_response = chat_bot.chat_predict(reference_response,user_input,new_node_id)  #用户输入，生成响应
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
                user_input = user.user_simulator.predict(input=agent_response, stop=['AI Assistant', 'user over'])
                # 用户模拟器输出文本清洗
                user_input = user_input.split('\n')[0]
                user_input = re.sub(r"\*\*User's question\*\*", '', user_input)
                new_node_id = agent_node_location_result.update_node_loc(user_input,agent_response,current_node_id)
                if toolgraph.raw_texts[str(new_node_id)] == "<end>":
                    break
                reference_response = toolgraph.raw_texts[str(new_node_id)]  # 获取跳转节点对应的文本内容
                turn_pre_agent["pre_doc_id"] = f"chart-{new_node_id}"
                agent_response = chat_bot.chat_predict(reference_response, user_input,new_node_id)
                # AI agent输出文本清洗
                agent_response = re.sub(r"<think>.*?</think>", "", agent_response, flags=re.DOTALL).strip()
                agent_response = agent_response.split('\n')[0]
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
                current_node_id = new_node_id
                if toolgraph.node_terminals[current_node_id] and len(toolgraph.edge_attr[current_node_id].items())==1:
                    break
                if turn_count>turn_budget:
                    break
        pre_dialog[dialogue_id]["utterences"]=utterences
        pre_dialog[dialogue_id]["flowchart"]=tool_name
        print("对话结束")

        save_dict_to_json(pre_dialog, str(results_file))
