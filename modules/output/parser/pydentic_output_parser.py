from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import Runnable


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY")
)

class Student(BaseModel):
    name:str = Field(...,description="name of the student")
    marks:int = Field(...,description="marks of the student")
    grade:Literal["A","B","C","D","F"] = Field(...,description="grade of the student")


template:PromptTemplate = ChatPromptTemplate.from_messages([
    ("system","You are a  best grading assistant. By watchign the list of json  data  will he pass or fail with name all  in the output you have to give it in the format  of list of {format_instructions}"),
    ("user", "text : {text}")
]) 


list_of_students:list[dict[str,str]] = [
    {
        "name":"john",
        "marks":34,
   
    },
    {
        "name":"Doe",
        "marks":45,
    
    },
    {
        "name":"jack",
        "marks":64,
    },
    {
        "name":"jhon",
        "marks":76,
    },
    {
        "name":"jhon",
        "marks":88,
    },
]

parser:PydanticOutputParser = PydanticOutputParser(pydantic_object=Student)

chain:Runnable[str] = template | model | parser

response = chain.invoke({"text":list_of_students[1],"format_instructions":parser.get_format_instructions()})

print(type(response),response)