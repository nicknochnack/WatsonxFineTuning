# Import IBMGen Library 
from genai.model import Credentials
from genai.schemas import GenerateParams, ModelType 
from genai.extensions.langchain import LangChainInterface
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import streamlit for the UI 
import streamlit as st

# Set your Model ID
MODEL_ID = 'YOUR PEFT MODEL ID HERE'
# Set your APIKEY
APIKEY = 'YOUR WATSONX API KEY HERE'
# Define credentials 
creds = Credentials(APIKEY)

with st.sidebar: 
    st.info('Set the sampling parameters') 
    decoding_type = st.radio('Set the Decoding Type', ['Greedy', 'Sample'])
     
    decoding_params = {}
    if decoding_type =='Sample': 
        decoding_params['max_new_tokens'] = int(st.slider('Select Max Tokens', 1, 100))
        decoding_params['min_new_tokens'] = int(st.slider('Select Min Tokens', 1, 100))
        decoding_params['temperature'] = st.slider('Select Temperature', 0.0, 2.0)
        decoding_params['top_p'] = st.slider('Select Top P',0.0, 1.0) 
        decoding_params['top_k'] = st.slider('Select Top K',1, 100) 
        decoding_params['repetition_penalty'] = st.slider('Select Repetition Penalty', 1.0,2.0)         


# Define generation parameters 
params = GenerateParams(decoding_method=decoding_type.lower(),
                        **decoding_params)

# Create an instance of the LLM
llm = LangChainInterface(model=MODEL_ID, params=params, credentials=creds)

# Title for the app
st.title('🤖 Prompting a PEFT LLM')
# Prompt box 
prompt = st.text_input('Enter your prompt here')
# If a user hits enter
if prompt: 
    # Pass the prompt to the llm
    response = llm(prompt)
    # Write the output to the screen
    st.write(response)