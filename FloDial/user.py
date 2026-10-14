from utils import *
from langchain.chains import LLMChain
import openai
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
import tenacity
import os
import re


TEMPLATE = '''You are a dialogue simulator acting as a user engaged in troubleshooting with an AI assistant. Your task is to generate the user’s reply at each turn, based on:
1. The prior information
2. The reference dialogue
3. Unanswered questions

Simulation Rules:
1. Your reply should reflect the **same polarity, intent, and information** as the user responses in the reference dialogue. Do not just reply a single word like yes or no.
2. Do not violate the prior information and reference dialogue.
3. You may ask clarifying questions, if such behavior appears in the reference dialogue (marked with the annotation **"User's question"** in the reference dialogue), but only within the scope and context of the reference dialogue — do not introduce unrelated or fabricated topics.
4. After asking the clarifying questions, remember to answer the unanswered questions.
5. If the question is new (not in the reference dialogue), infer a plausible reply based on the known context and reference dialogue style.
6. Only generate the user’s reply. Do not include internal thoughts or commentary.

Use these to match the expected tone, structure, and behavior in your replies.

Prior information:
{prio_information}

Reference dialogue:
{ref_dialog}

Current conversation:
{{history}}

Unanswered questions：
{{unanswerd_question}}

AI Assistant: {{input}}

User:
'''

def prepare_user_simulator(dialog,chart_data,chart_dict):
    ref_dialog = []

    for i,turn in enumerate(dialog['utterences']):
        role = 'User' if i % 2 == 0 else 'AI Assistant'
        if role=='AI Assistant' and i==1:
            if turn['grounded_doc_id']!="chart-0":
                prio_info = []
                num = (re.search(r'\d+', turn['grounded_doc_id'])).group(0)
                index = list(chart_dict['nodes'].keys()).index(num)

                prio_info_list = chart_data['paths_text'][index]
                for ind,text in enumerate(prio_info_list):
                    que='Question' if ind%2==0 else 'Answer'
                    utter = que+': '+text
                    prio_info.append(utter)
                prio_info = '\n'.join(prio_info)
        if role == 'AI Assistant' and turn.get('grounded_doc_id', '').startswith('faq'):
            ref_dialog[i-1]= "(**User's question**)"+ ref_dialog[i-1]
        utter = role + ': ' + turn['utterance']
        ref_dialog.append(utter)
    end_dialog = ref_dialog[-4:]
    end_dialog = '\n'.join(end_dialog)
    ref_dialog = '\n'.join(ref_dialog)

    fisrt_user_utter = dialog['utterences'][0]['utterance']
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError('DEEPSEEK_API_KEY environment variable is required')
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=api_key)
    template = TEMPLATE.format(
        ref_dialog=ref_dialog,
        prio_information = prio_info
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "unanswerd_question"]
    )

    class ConversationBufferMemoryStrip(ConversationBufferMemory):
        def _get_input_output(self, inputs, outputs):
            input_str, output_str = super()._get_input_output(inputs, outputs)
            return input_str.strip(), output_str.strip()

    memory = ConversationBufferMemoryStrip(
        human_prefix='AI Assistant',
        ai_prefix='User',
        memory_key='history',
        input_key='input'
    )
    user = LLMChain(
        prompt=prompt,
        llm=llm,
        memory=memory
    )

    return user, fisrt_user_utter

class User:

    def __init__(self, dialog,chart_data,chart_dict):
        self.user_simulator, self.fisrt_user_utter = prepare_user_simulator(dialog,chart_data,chart_dict)
        self.turn_idx = 0

    def __call__(self, sys_utter, callbacks=None):
        self.turn_idx += 1

        if self.turn_idx == 1:  # First user utter: use fixed instead AI generated
            assert sys_utter in ['', None]
            self.user_simulator.memory.chat_memory.add_ai_message(self.fisrt_user_utter)
            return self.fisrt_user_utter

        retrying = tenacity.Retrying(wait=tenacity.wait_exponential(min=2, max=60),
                                     stop=tenacity.stop_after_attempt(8),
                                     before_sleep=tenacity_retry_log,
                                     retry=tenacity.retry_if_exception_type(openai.OpenAIError))
        user_utter = retrying(self.user_simulator.predict, input=sys_utter, stop='AI Assistant', callbacks=callbacks)

        user_utter = user_utter.strip()
        return user_utter
