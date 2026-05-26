from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant that can answer questions and help with tasks. if you don't know the answer, just say that you don't know. Do not make up an answer or give false information."),
    ("user", "explain the following code: {input}"),
])

 