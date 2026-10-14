from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_community.chat_models import ChatTongyi
from langchain_ollama import OllamaLLM

TEMPLATE = '''You are an AI assistant based on flowchart.

you will be provided with the user's latest input, the conversation history, and a reference response for the current turn. 

The reference response is the intended message. You must preserve its original intent, sentence type (e.g., question vs. statement), and all domain-specific or technical terms. Revise only for minor improvements in grammar, fluency, or conversational flow — do not paraphrase or reframe the meaning.

Keep your responses concise and focused. Do not generate explanations, thoughts, or extra commentary — output only the agent’s reply.

Conversation history:
{{history}}

User input:
{{input}}

Reference response:
{reference_response}

AI assistant:
'''


class ConversationBufferMemoryStrip(ConversationBufferMemory):
    def _get_input_output(self, inputs, outputs):
        input_str, output_str = super()._get_input_output(inputs, outputs)
        return input_str.strip(), output_str.strip()

class agent_chat:
    def __init__(self,Model_NAME,OPENROUTER_BASE_URL,api_key):
        self.Model_NAME = Model_NAME
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
        else:
            llm = OllamaLLM(model=self.Model_NAME)

        self.llm = llm
        return llm
    def chat_predict(self,reference_response,user_utter):
        template = self.update_template(reference_response)
        agent = LLMChain(
            prompt=PromptTemplate.from_template(template),
            llm=self.llm,
            memory=self.memory
        )
        agent_response = agent.predict(input=user_utter)
        return agent_response
    def update_template(self,reference_responce):
        template = TEMPLATE.format(
            reference_response=reference_responce
        )
        return template
