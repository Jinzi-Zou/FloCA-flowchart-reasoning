from utils import *
from langchain.chains import LLMChain
import openai
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
import tenacity
import os

TEMPLATE = '''You are a dialogue simulator acting as a user interact with an AI assistant. Your task is to generate the user’s reply at each turn, based on the reference dialogue.

Simulation Rules:
1. Your reply should reflect the **same polarity, intent, and information** as the user responses in the reference dialogue. Do not just reply a single word like yes or no.
2. Do not violate the reference dialogue.
3. If the question from AI assistant is new (not in the reference dialogue), infer a plausible reply based on the known context and reference dialogue style.
4. Only generate the user’s reply. Do not include internal thoughts or commentary.
5. If the reference dialogue contains the same agent question appearing multiple times, it indicates a loop. In this case, when generating the user reply, you MUST advance the loop by using the corresponding user reply for the current loop occurrence (1st time → 1st matching reply in reference, 2nd time → 2nd matching reply, etc.), instead of repeating the first loop answer.
6. You MUST align your reply to the CURRENT AI Assistant utterance. Only respond to what is asked/required in the current turn. Never reveal information that belongs to a later assistant question in the reference dialogue (no “answering the next question early”).

Use these to match the expected tone, structure, and behavior in your replies.

Reference dialogue:
{ref_dialog}

Current conversation:
{{history}}

AI Assistant: {{input}}

User:
'''


def prepare_user_simulator(dialog):
    # Prepare Reference Dailogue
    ref_dialog = []

    for i,turn in enumerate(dialog['utterences']):
        role = 'User' if i % 2 == 0 else 'AI Assistant'
        utter = role + ': ' + turn['utterance']
        ref_dialog.append(utter)
    ref_dialog = '\n'.join(ref_dialog)
    # First User Utter
    fisrt_user_utter = dialog['utterences'][0]['utterance']
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError('DEEPSEEK_API_KEY environment variable is required')
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0, api_key=api_key)
    # Prompt & Chain
    template = TEMPLATE.format(
        ref_dialog=ref_dialog
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["input"]
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
        # verbose=True
    )

    return user, fisrt_user_utter



class User:

    def __init__(self, dialog):
        self.user_simulator, self.fisrt_user_utter = prepare_user_simulator(dialog)
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
