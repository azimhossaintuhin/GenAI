from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY")
)


template:PromptTemplate = ChatPromptTemplate.from_messages([
    ("system","You are a  best grading assistant. By watchign the list of  json  data  will he pass or fail with name {format_instructions}"),
    ("user", "text : {text}")
]) 

data = [
    {
        "name":"john",
        "marks":34
    },
    {
        "name":"Doe",
        "marks":45
    },
    {
        "name":"jack",
        "marks":64
    },
    {
        "name":"jhon",
        "marks":76
    },
    {
        "name":"jhon",
        "marks":88
    },
]


parser:JsonOutputParser = JsonOutputParser()

chain:Runnable[str] = template | model | parser

response = chain.invoke({"text":data,"format_instructions":parser.get_format_instructions()})

print(type(response),response)
