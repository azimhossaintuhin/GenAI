from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY"))

prompt:PromptTemplate = PromptTemplate(
    template = """ 
    Good Morning {name}.
    write a poem in {lang} language.
    """,
    input_variables=["name","lang"],
    validate_template=True,
)

parser:StrOutputParser = StrOutputParser()

chain:Runnable[str,str] = prompt | model | parser

response = chain.invoke({"name":"John","lang":"English"})

chain.get_graph().print_ascii()