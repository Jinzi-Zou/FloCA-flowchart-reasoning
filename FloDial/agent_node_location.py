from utils import *
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
import re


TEMPLATE_first_node = '''You are an AI assistant. Your task is to determine the next node the correct answer to the given question based on the user's input.

There are only three possible answers:
- yes: The user's input directly and unambiguously affirms the question.
- no: The user's input directly and unambiguously denies the question.
- trigger: In all other cases, including vague, indirect, incomplete, or unrelated responses, you must choose trigger.

Important rules:
- Only output yes or no if the user’s input is a clear, direct answer to the question.
- Do not infer or over-interpret. If the answer is not obvious, always choose trigger.
- Do not rely solely on the presence of positive or negative words. The input must explicitly match the intent of the question.

Output only one of the following words: **yes**, **no**, or **trigger**. Do not provide any explanation.

Given question:
{question}

User input:
{input}
'''

TEMPLATE_update_node = '''You are an assistant that determines whether the user answered a given yes/no question.

Your job is to analyze the user's input and output one of the following:
- "yes" — if the user clearly or implicitly answered "yes"
- "no" — if the user clearly or implicitly answered "no"
- "trigger" — Need to trigger a knowledge base to answer user's question. if the user did NOT answer the question and instead asked a new question or changed the topic.

You MUST output only one word:  
**yes**, **no**, or **trigger**

You must carefully analyze what the question is asking, do not rely solely on the presence of positive or negative words in user input.

Do not include any punctuation or extra words.

Given question:  
{question} 

User input:  
{input}    
'''

class agent_node_location:
    def __init__(self, tool,Model_NAME,OPENROUTER_BASE_URL,api_key):
        self.Model_NAME = Model_NAME
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.api_key = api_key
        self.tool = tool
        self.first_node_agent = self.prepare_first_node_loc()
        self.update_node_agent  = self.prepare_update_node_loc()

    def prepare_first_node_loc(self):
        if self.Model_NAME=="deepseek-reasoner":
            llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=self.api_key)
        elif self.Model_NAME in {"openai/gpt-4o", "openai/gpt-5","anthropic/claude-opus-4.1"}:
            llm = ChatOpenAI(
                model_name=self.Model_NAME,
                base_url=self.OPENROUTER_BASE_URL,
                temperature=0,
                api_key=self.api_key,
                timeout=10,
            )
        elif self.Model_NAME == 'qwen-plus':
            llm = ChatTongyi(model="qwen-plus", temperature=0, api_key=self.api_key)
        else:
            llm = OllamaLLM(model=self.Model_NAME)

        agent = LLMChain(
            prompt=PromptTemplate.from_template(TEMPLATE_first_node),
            llm=llm
        )
        return agent

    def prepare_update_node_loc(self):
        if self.Model_NAME == "deepseek-reasoner":
            llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=self.api_key)
        elif self.Model_NAME in {"openai/gpt-4o", "openai/gpt-5","anthropic/claude-opus-4.1"}:
            llm = ChatOpenAI(
                model_name=self.Model_NAME,
                base_url=self.OPENROUTER_BASE_URL,
                temperature=0,
                api_key=self.api_key,
                timeout=10,
            )
        elif self.Model_NAME == 'qwen-plus':
            llm = ChatTongyi(model="qwen-plus", temperature=0, api_key=self.api_key)
        else:
            llm = OllamaLLM(model=self.Model_NAME)
        agent = LLMChain(
            prompt=PromptTemplate.from_template(TEMPLATE_update_node),
            llm=llm
        )
        return agent

    def first_node_loc(self,user_utter):
        current_node_id = 1

        while True:
            # 获取当前节点的文本
            current_node_text = self.tool.raw_texts[current_node_id]

            # 使用 LLM 用户模拟器生成用户回答
            user_response = self.first_node_agent.predict(input=user_utter, question=current_node_text)
            user_response = re.sub(r"<think>.*?</think>", "", user_response, flags=re.DOTALL).strip()
            # 判断是否能够跳转到下一个节点
            if "trigger" in user_response.lower():
                print(f"Stopping at Node ID: {current_node_id}")
                return current_node_id  # 停止流程，返回当前节点 ID

            # 根据用户模拟器的回答选择边
            if "yes" in user_response.lower():
                edge_value = 1
            elif "no" in user_response.lower():
                edge_value = 0
            else:
                print(f"Invalid response: {user_response}. Stopping at Node ID: {current_node_id}")
                return current_node_id

            # 查找下一节点
            next_node_id = find_next_node(self.tool, current_node_id, edge_value)
            if self.tool.node_terminals[next_node_id]:
                return current_node_id

            if next_node_id is None:
                print(f"No further nodes. Stopping at Node ID: {current_node_id}")
                return current_node_id

            # 更新当前节点
            current_node_id = next_node_id

    def update_node_loc(self, user_utter,agent_response,current_node_id,QA_tool):
        # 使用 LLM 用户模拟器生成用户回答
        user_response = self.update_node_agent.predict(input=user_utter, question=agent_response)
        user_response = re.sub(r"<think>.*?</think>", "", user_response, flags=re.DOTALL).strip()
        # 判断是否能够跳转到下一个节点
        if "trigger" in user_response.lower():
            return -1  # 停止流程，返回当前节点 ID

        # 根据用户模拟器的回答选择边
        if "yes" in user_response.lower():
            edge_value = 1
        elif "no" in user_response.lower():
            edge_value = 0
        else:
            print(f"Invalid response: {user_response}. Stopping at Node ID: {current_node_id}")
            return -1

        # 查找下一节点
        next_node_id = find_next_node(self.tool, current_node_id, edge_value)

        if next_node_id is None:
            print(f"No further nodes. Stopping at Node ID: {current_node_id}")
            return current_node_id
        else:
            return next_node_id