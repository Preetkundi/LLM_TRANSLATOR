import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LangChain model
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
parser = StrOutputParser()

# Chain
chain = prompt_template | model | parser

# Streamlit UI
st.set_page_config(page_title="LangChain Translator", layout="centered")
st.title("🈂️ LangChain Translator (Gemma 2b)")

# Input fields
language = st.text_input("Target Language", value="French")
text = st.text_area("Text to Translate", value="Hello, how are you?")

if st.button("Translate"):
    with st.spinner("Translating..."):
        try:
            output = chain.invoke({"language": language, "text": text})
            st.success("Translation complete:")
            st.text_area("Output", value=output, height=150)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
