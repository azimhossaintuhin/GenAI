from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from  langchain_google_genai import ChatGoogleGenerativeAI

from decouple import config



model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))


messages = [
    SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks. if you don't k now the answer, just say that you don't know. Do not make up an answer or give false information.")    
]

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content[0].get("text","No response")))
    print(response.content[0].get("text","No response"))
    print("--------------------------------")
    
    print(messages)