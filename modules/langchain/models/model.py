from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))
response = model.invoke("Hello, are you up?")
print(response.content[0].get("text","No response"))
