from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableLambda
from decouple import config
from pydantic import BaseModel, Field
    

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=config("GOOGLE_API_KEY"),
    temperature=0.5,
)


class UserInfo(BaseModel):
    name: str
    age: int
    nationality: str
    marital_status: bool


parser = PydanticOutputParser(pydantic_object=UserInfo)


prompt1 = PromptTemplate(
    template="""
Extract user information.

{format_instructions}

Text:
{user_info}
""",
    input_variables=["user_info"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


prompt2 = PromptTemplate(
    template="Write a short summary on:\n{profile}",
    input_variables=["profile"]
)


profile_chain = prompt1 | model | parser


runnable = RunnableParallel(
    {
        "original": RunnablePassthrough(),
        "profile": profile_chain,
         "length    ": RunnableLambda(lambda x: len(x))
    }
)


user_information = """
Hi my name is Tuhin , I am 16 years old ,
I am Bangladeshi, Im not married.
"""

res = runnable.invoke(user_information)

print(res)