from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence
from decouple import config


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = config("GOOGLE_API_KEY"),
    temperature = 0.5,
)

parser  = StrOutputParser()


prompt1 = PromptTemplate(
    template = "Write a  post based on {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Create a pool {topic}",
    input_variables = ["topic"]
)

seq= RunnableSequence(
    prompt1 ,
    model ,
    parser
)

seq2 = RunnableSequence(
    prompt2 ,
    model ,
    parser
)

perallal  = RunnableParallel({
    "post" : seq,
    "poetry" : seq2
}
)

response = perallal.invoke({"topic":"Electricity"})
print(response["post"])
print("\n"*5)
print(response["poetry"])
