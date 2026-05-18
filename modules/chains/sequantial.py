from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable , RunnableLambda
from langchain_core.output_parsers import StrOutputParser


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY"))


prompt1: PromptTemplate = PromptTemplate(
    input_variables=["lang"],
    template="""
    Good Morning {lang}.
    write a poem in {lang} language.
    """,
    validate_template=True,
)

prompt2: PromptTemplate = PromptTemplate(
    input_variables=["review", "lang"],
    template="""
    Good Morning {review}.
    write a short summary of the following {review} in {lang} language.
    """,
    validate_template=True,
)

parser = StrOutputParser()


chain = prompt1 | model | parser | RunnableLambda(lambda x: {"review": x, "lang": "English"}) | prompt2 | model | parser

response = chain.invoke({"lang":"English","review":"write a short summary of the following review in {lang} language."})

print(response)