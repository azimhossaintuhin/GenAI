from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser ,  ResponseSchema
from langchain_core.runnables import Runnable


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY")
)

schema = [
    ResponseSchema(name="name",description="name of the student",required=True),
    ResponseSchema(name="marks",description="marks of the student",required=True),
    ResponseSchema(name="grade",description="grade of the student",required=True),
]

template:PromptTemplate = ChatPromptTemplate.from_messages([
    ("system","You are a  best grading assistant. By watchign the list of  json  data  will he pass or fail with name {format_instructions}"),
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

parser:StructuredOutputParser = StructuredOutputParser.from_response_schemas(schema)

chain:Runnable[str] = template | model | parser

response = chain.invoke({"text":list_of_students[0],"format_instructions":parser.get_format_instructions()})

print(type(response),response)