from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))

chat_history = []
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(response.content[0].get("text","No response"))
    print(response.content[0].get("text","No response"))
 
print(chat_history)