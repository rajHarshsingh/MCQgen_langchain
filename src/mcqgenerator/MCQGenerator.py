import os
import json
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=openai_key, 
                 model_name='gpt-3.5-turbo',
                 temperature=0.8)

TEMPLATE="""
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON and use it as a guide.\
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=['text','number','subject','tone','response_json']
)

quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_generation_prompt,
    output_key='quiz',
    verbose=True
)

# a second task will be setup to review the questions framed
TEMPLATE2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for the analysis.\
If the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changes and change the tone such that it perfectly fits the student abilities.
Quiz MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    template=TEMPLATE2,
    input_variables=['subject','quiz']
)

review_chain = LLMChain(
    llm=llm,
    prompt=quiz_evaluation_prompt,
    output_key='review',
    verbose=True
)

# combining both the chains using SequentialChain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=['text','number','subject','tone','response_json'],
    output_variables=['quiz','review'],
    verbose=True
)