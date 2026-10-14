from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi

TEMPLATE = '''You are an AI assistant. Your task is to answer the user's question based on a predefined set of question-and-answer pairs.

The template below defines standard answers for specific questions.

- If the user's question matches one of the predefined questions, extract and return the corresponding standard answer exactly as provided. Do not rephrase, summarize, or add new information in the answer.

- If no appropriate match is found, generate a direct reply to the user's input without any extra explanation. Your response should follow the same language style as the answers in the template — clear, concise, and to the point.

Question-and-answer template:
{qapair}

Conversation:
Agent_response: {{agent_response}}
User: {{input}}

AI Assistant:
'''


class agent_qa:
    def __init__(self,Model_NAME,OPENROUTER_BASE_URL,api_key):
        # pass
        self.Model_NAME = Model_NAME
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.api_key = api_key

    def prepare_agent_qa(self,qatool):
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
        lines = []
        for qa in qatool:
            for k, v in qa.items():
                if k == 'id':
                    break
                if k == 'q':
                    k = 'Question'
                elif k == 'a':
                    k = 'Answer'
                lines.append(f"{k}:{v}")
            lines.append("")  # 添加一个空行作为段落分隔

        # 最终拼接成字符串
        result = '\n'.join(lines)
        # Prompt & Chain
        template = TEMPLATE.format(
            qapair=result,
        )
        agent = LLMChain(
            prompt=PromptTemplate.from_template(template),
            llm=llm
        )
        return agent