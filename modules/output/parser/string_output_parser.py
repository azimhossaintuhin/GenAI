from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable


model:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",api_key=config("GOOGLE_API_KEY"))
parser:StrOutputParser = StrOutputParser()

prompt1:PromptTemplate = PromptTemplate(
    template = """ wirte me a verbose explanation  of the following {topic} """,
    input_variables=["topic"],
    validate_template=True, 
)


prompt2:PromptTemplate = PromptTemplate(
    template = """ make a short summary  of the following {text} """,
    input_variables=["text"],
    validate_template=True,
)

chain:Runnable[str] = prompt1 | model | parser | prompt2 | model | parser


print(chain.invoke({"topic":"Python Lists"}))

