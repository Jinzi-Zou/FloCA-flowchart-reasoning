from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_ollama import OllamaLLM
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
import difflib
import re
import torch


SYSTEM_RULES_first = '''
You are an AI assistant to determine the flowchart node of current dialog turn. Your task is to determine whether the current dialog can transition to a next node
by matching the user's input with one of the outgoing edge attributes of the current node. the edge attribute can be infered by user input if need to navigate to next node.

Transition Rules：
1. Each outgoing edge has a textual attribute.
2. If Edge attribute only contain "_", then directly output the special token: trigger and stop reasoning.
3. You may transition to the next node if the user's input expresses the same meaning as exactly ONE edge attribute.
4. If none of the edge attributes are matched by the user's input, you MUST NOT transition.


Output Rules：
If a valid transition exists: output ONLY the matched edge attribute (exact string).
Otherwise: Output ONLY the special token: trigger

Do NOT explain your reasoning.
Do NOT output anything other than the required string.
'''
TEMPLATE_first_node = '''
Edge attribute:
{nextattr}

Current node text:
{question}

User input:
{input}
'''

SYSTEM_RULES_UPDATE = '''
You are an AI assistant to determine the flowchart node of current dialog turn. Your task is to determine whether the current dialog can transition to a next node
by matching the user's input with one of the outgoing edge attributes of the current node. the edge attribute can be infered by user input if need to navigate to next node.

Transition Rules：
1. Each outgoing edge has a textual attribute.
2. Edge attribute "_" normally means the edge has no explicit textual label. However, if "_" appears together with exactly one other explicit attribute (e.g., "yes") as the two outgoing edges of the SAME decision/condition, then treat "_" as the implicit complementary branch (i.e., the opposite of that explicit attribute). For example, if the two attributes are ("yes", "_"), then "_" means "no"; if they are ("no", "_"), then "_" means "yes".
3. You may transition to the next node if the user's input expresses the same meaning as exactly ONE edge attribute.

Output Rules：
1. If Edge attribute only contain "_", then directly output the "_" and stop reasoning.
2. elIf a valid transition exists: output ONLY the matched edge attribute (exact string).
3. Otherwise: Output ONLY the special token: trigger

Do NOT explain your reasoning.
Do NOT output anything other than the required string.
'''

TEMPLATE_update_node = '''
Edge attribute:
{nextattr}

Current node text:
{question}

User input:
{input}
'''

def _norm(s: str) -> str:
    """Normalize for robust matching: strip, lower, remove trailing punctuation."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.strip('\'"`')
    # remove trailing punctuations like ".", "。", "!", "?" ...
    s = re.sub(r"[ \t\r\n]*[.。!！?？,，;；:：]+$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def match_edge_attr(user_response: str, next_attr: list[str]):

    # map normalized attr -> original attr (keep first)
    if user_response in next_attr:
        return user_response
    else:
        ur = _norm(user_response)
        if ur == "trigger" or ur.startswith("trigger ") or ur.startswith("trigger:"):
            return "trigger"

        # --- build normalized map for attrs ---
        norm2orig = {}
        for a in next_attr:
            na = _norm(a)
            norm2orig.setdefault(na, a)  # keep first if collision

        # --- 1) exact match after normalization ---
        if ur in norm2orig:
            return norm2orig[ur]

        # --- 2) prefix match (optional but useful): "yes i will" -> "yes" ---
        for na, orig in norm2orig.items():
            if na and ur.startswith(na):
                return orig

        # --- 3) fuzzy match fallback ---
        if norm2orig:
            cands = list(norm2orig.keys())
            best = difflib.get_close_matches(ur, cands, n=1, cutoff=0.85)
            if best:
                return norm2orig[best[0]]

        return None

class agent_node_location:
    def __init__(self, tool,Model_NAME,OPENROUTER_BASE_URL,api_key):
        self.Model_NAME = Model_NAME
        self.model_path = f'./models/{Model_NAME}'
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
        elif self.Model_NAME in {'Llama-3.1-8B-Instruct','Qwen2.5-7B-Instruct'}:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            llm = Llama31ChatLLM(model=model, tokenizer=tokenizer, max_new_tokens=16,default_system_rules=SYSTEM_RULES_first)
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
        elif self.Model_NAME in {'Llama-3.1-8B-Instruct','Qwen2.5-7B-Instruct'}:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            llm = Llama31ChatLLM(model=model, tokenizer=tokenizer, max_new_tokens=16,default_system_rules=SYSTEM_RULES_UPDATE)
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
            current_node_text = self.tool.raw_texts[str(current_node_id)]
            next_attr = []
            for edge_attr, node_id in self.tool.edge_attr[f'{current_node_id}'].items():
                next_attr.append(edge_attr)
            str_next_attr = "\n".join(next_attr)
            # 使用 LLM 用户模拟器生成用户回答
            user_response = self.first_node_agent.predict(input=user_utter, question=current_node_text,nextattr=str_next_attr)
            user_response = re.sub(r"<think>.*?</think>", "", user_response, flags=re.DOTALL).strip()
            # 判断是否能够跳转到下一个节点
            if next_attr == ["_"]:
                return current_node_id
            user_response = match_edge_attr(user_response, next_attr)
            if user_response is not None:
                if "trigger" in user_response.lower():
                    print(f"Stopping at Node ID: {current_node_id}")
                    return current_node_id  # 停止流程，返回当前节点 ID
            # 查找下一节点
                else:
                    next_node_id = self.tool.edge_attr[f'{str(current_node_id)}'][f'{user_response}']
                    return next_node_id
            else:
                print(f"Invalid response: {user_response}. Stopping at Node ID: {current_node_id}")
                return current_node_id


    def update_node_loc(self, user_utter,agent_response,current_node_id):
        current_node_text = self.tool.raw_texts[str(current_node_id)]
        next_attr = []
        for edge_attr, node_id in self.tool.edge_attr[f'{current_node_id}'].items():
            next_attr.append(edge_attr)
        str_next_attr = "\n".join(next_attr)
        # 使用 LLM 用户模拟器生成用户回答
        user_response = self.update_node_agent.predict(input=user_utter, question=current_node_text, nextattr=str_next_attr)
        user_response = re.sub(r"<think>.*?</think>", "", user_response, flags=re.DOTALL).strip()
        # 判断是否能够跳转到下一个节点
        user_response = match_edge_attr(user_response, next_attr)
        if user_response is not None:
            if user_response != "trigger":
                next_node_id = self.tool.edge_attr[f'{str(current_node_id)}'][f'{user_response}']
                return next_node_id
            elif user_response == "trigger":
                if next_attr == ['_']:
                    next_node_id = self.tool.edge_attr[f'{str(current_node_id)}']['_']
                    return next_node_id
                else:
                    return current_node_id
        elif next_attr==['_']:
            next_node_id = self.tool.edge_attr[f'{str(current_node_id)}']['_']
            return next_node_id
        else:
            print(f"Invalid response: {user_response}. Stopping at Node ID: {current_node_id}")
            return current_node_id


class Llama31ChatLLM(LLM):
    model: Any
    tokenizer: Any
    max_new_tokens: int = 32
    default_system_rules:str

    @property
    def _llm_type(self) -> str:
        return "llama31_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": self.default_system_rules},
            {"role": "user", "content": prompt}
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = out[0, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        text = text.strip().splitlines()[0].strip()
        return text
