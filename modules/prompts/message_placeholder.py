from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import BaseMessage
from typing import List

# from langchain_core.messages import AIMessage


model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))

chat_template:ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant that can answer questions and help with tasks. if you don't know the answer, just say that you don't know. Do not make up an answer or give false information."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "explain the following code: {input}"),
    
])

chat_history:List[BaseMessage] = []

with open("chat_history.txt") as  f:
    for line in f:
        chat_history.extend(f.readlines())

# create  prompt 

while True:
    query = input("User : ")
    if "exit" == query.lower():
        break
    
    prompt:ChatPromptTemplate = chat_template.invoke({
        "chat_history":chat_history,
        "input":query
    })

    res = model.invoke(prompt)
 
    print(res.content[0].get("text","No response"))

