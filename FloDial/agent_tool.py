from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM


TEMPLATE = '''You are an AI assistant. Your task is to understand the user's current problem and select the correct API from the list below to address it.

Use the textual descriptions of each API to determine which one is most appropriate for resolving the issue.

You must choose only one API from the provided list — do not invent or reference any APIs beyond this scope. Respond ONLY with the APIs' name. Do NOT include any explanations.

APIs' description:
{tool_disc}

Available APIs:
{domain}

User input:{{input}}

AI Assistant:
'''


class agent_toolselction:
    def __init__(self, tool_disc,domain,Model_NAME,OPENROUTER_BASE_URL,api_key):
        self.Model_NAME = Model_NAME
        self.OPENROUTER_BASE_URL = OPENROUTER_BASE_URL
        self.api_key = api_key
        self.agent_toolselection = self.prepare_agent_toolselect(tool_disc,domain)


    def prepare_agent_toolselect(self,tool_disc, domain):
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
        domain_str = ', '.join(domain)
        tool_disc_str = ''.join([f"{k}: {v}\n" for k, v in tool_disc.items()])
        # Prompt & Chain
        template = TEMPLATE.format(
            tool_disc=tool_disc_str,
            domain=domain_str
        )
        agent = LLMChain(
            prompt=PromptTemplate.from_template(template),
            llm=llm
        )
        return agent