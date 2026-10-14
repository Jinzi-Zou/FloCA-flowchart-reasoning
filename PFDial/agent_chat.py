from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
from langchain_ollama import OllamaLLM
import torch


SYSTEM_RULES = '''You are an AI assistant based on a flowchart-driven dialog system. Your ONLY task is to lightly polish the reference response so that it sounds
more natural and fluent to a human user.

You will be provided with:
- the user's latest input,
- the conversation history,
- a reference response for the current turn,
- and the next edge attributes from the current flowchart node.

Response Generation Rules：
1. The reference response defines the intended meaning of this turn.
   You MUST preserve:
   - the original intent and meaning,
   - all domain-specific or technical terms.
2.You may revise the reference response for:
   - grammar,
   - fluency,
   - and natural conversational flow.
   These edits must be surface-level only.
   Do NOT change, extend, or enrich the content.
3. ONLY if the reference response is explicitly marked as:
- “循环判断”, or
- “条件判断”
you MUST ask the user a question that:
- Is directly based on the reference response intent.
- Naturally guides the conversation forward.
- Implicitly or explicitly reflects the available next edge attributes.
- Sounds like normal human conversation, NOT a system instruction.
(e.g., 条件判断：是否需要导游服务?),
rewrite it as "请问您是否需要导游服务？"
4. If the reference response is an impersonal or system-style description
(e.g., “顾客选择旅游套餐”),
rewrite it as a first-person assistant utterance
(e.g., “好的，我正在为您查找可选择的旅游套餐。”).
5. The output language MUST be consistent with the language of the reference response:
   - If the reference response is in Chinese, output in Chinese.
   - If the reference response is in English, output in English.
   - Do NOT translate or mix languages.
'''

TEMPLATE = '''
Conversation history:
{{history}}

User input:
{{input}}

Reference response:
{reference_response}

Next edge attributes:
{nextedgeattr}
'''


class ConversationBufferMemoryStrip(ConversationBufferMemory):
    def _get_input_output(self, inputs, outputs):
        input_str, output_str = super()._get_input_output(inputs, outputs)
        return input_str.strip(), output_str.strip()

class agent_chat:
    def __init__(self,tool,Model_NAME,OPENROUTER_BASE_URL,api_key):
        self.tool=tool
        self.Model_NAME = Model_NAME
        self.model_path = f'./models/{Model_NAME}'
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.api_key = api_key
        self.memory = ConversationBufferMemoryStrip(
            human_prefix='User',
            ai_prefix='AI Assistant',
            memory_key='history'
        )
    def prepare_agent_chat(self):
        if self.Model_NAME == "deepseek-reasoner":
            llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=self.api_key)
        elif self.Model_NAME in {"openai/gpt-4o","openai/gpt-5","anthropic/claude-opus-4.1"}:
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
            llm = Llama31ChatLLM(model=model, tokenizer=tokenizer, max_new_tokens=512)
        else:
            llm = OllamaLLM(model=self.Model_NAME)

        self.llm = llm
        return llm
    def chat_predict(self,reference_response,user_utter,current_node_id):
        next_attr = []
        for edge_attr, node_id in self.tool.edge_attr[f'{current_node_id}'].items():
            next_attr.append(edge_attr)
        str_next_attr = "\n".join(next_attr)
        if self.tool.node_loop[f'{current_node_id}']:
            reference_response = f'循环判断-{reference_response}'
        elif self.tool.node_condition[f'{current_node_id}']:
            reference_response = f'条件判断-{reference_response}'
        template = self.update_template(reference_response,str_next_attr)
        agent = LLMChain(
            prompt=PromptTemplate.from_template(template),
            llm=self.llm,
            memory=self.memory
        )
        agent_response = agent.predict(input=user_utter)
        return agent_response
    def update_template(self,reference_responce,str_next_attr):
        template = TEMPLATE.format(
            reference_response=reference_responce,
            nextedgeattr=str_next_attr
        )
        return template

class Llama31ChatLLM(LLM):
    model: Any
    tokenizer: Any
    max_new_tokens: int = 32

    @property
    def _llm_type(self) -> str:
        return "llama31_chat"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_RULES},
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
