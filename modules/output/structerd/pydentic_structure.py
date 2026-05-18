from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from pydantic import BaseModel,Field
from typing import List,Optional,Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


model =  ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",api_key=config("GOOGLE_API_KEY"))


# Schema 
class Review(BaseModel):
    review : int = Field(...,ge=1,le=5 , description="This should be the rating of the text in the range of 1-5")
    summary : str = Field(...,description="the summary of the review")
    sentiment :Literal["positive","negative"] = Field(...,description="the sentiment of the text must be either positive or negative")
    return_likely : Literal["likely","not likely"] = Field(...,description="the user is likely to return or not likely to return")
    pros : List[str] = Field(...,description="the pros of the text")
    cons : List[str] = Field(...,description="the cons of the text")
    suggested_changes : List[str] = Field(...,description="the suggested changes of the text")
    customer_suggestion: Optional[str] = Field(None,description="the suggestion of the customer") 
    language : str = Field(...,description="the language of the text")
    traslated_text : Optional[str] = Field(None,description="the traslated text in english if you recived the text in any other language ")
    will_customer_repeat: Literal["likely","not likely"] = Field(...,description="the user is likely to return or not likely to return")

model = model.with_structured_output(Review)

template:ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system","""
    You are a expert in language and translation . your only job is to translate the following text into english. 
    the text will be in any language, but you must translate it into english. 
    """),
    ("user", "review : {review}")
])

reviews:str = input("Enter your review : ")

chain:Runnable[str,Review] = template | model

final_result:Optional[Review] = chain.invoke({
    "review":reviews,
})


print("---------------------")
print("Customer Review")
print("---------------------")
print("Review : ",final_result.review)
print("Summary : ",final_result.summary)
print("Sentiment : ",final_result.sentiment)
print("Likelihood of user returning to shop : ",final_result.return_likely)
print("Pros : ",final_result.pros)
print("Cons : ",final_result.cons)
print("Suggested Changes : ",final_result.suggested_changes)
if final_result.customer_suggestion:
    print("Customer Suggestion : ",final_result.customer_suggestion)
print("Language : ",final_result.language)
if final_result.traslated_text:
    print("Translated Text : ",final_result.traslated_text)
print("Will Customer Repeat : ",final_result.will_customer_repeat)
print("---------------------")