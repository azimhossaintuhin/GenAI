from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from decouple import config


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = config("GOOGLE_API_KEY"),
    temperature = 0.5,
)

parser  = StrOutputParser()

prompt = PromptTemplate(
    template = "Write a  post based on {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "now wirte  a  short summary  on \n {text}",
    input_variables = ["text"]
)

chain = RunnableSequence(
    prompt , model , parser ,  prompt2 , model , parser
)

response = chain.invoke({"topic":"Electricity"})
print(response)