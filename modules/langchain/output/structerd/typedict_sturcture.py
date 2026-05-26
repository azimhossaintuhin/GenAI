from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from typing import TypedDict, Optional,Annotated,Literal
from langchain_core.prompts import ChatPromptTemplate


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY") ,temperature=0.2)

# Schema
class Review(TypedDict):
    review: Annotated[int,"How much star you would give it based on review"]
    sentiment: Annotated[Literal["positive", "negative"],"the text sentiment must be positive , negative only"]
    return_likely: Annotated[Literal["likely", "not likely"],"the text sentiment must be   likely or   not likely only"]
    pros : Annotated[list[str],"the pros must be in the list of strings"]
    cons : Annotated[list[str],"the cons must be in the list of strings"]
    suggested_changes: Annotated[list[str],"the suggested changes must be in the list of strings"]
    
template = ChatPromptTemplate.from_messages([
    ("system","You are a  customer senitment analyzer that analyze the sentiment of the following text and return the sentiment of the text.  the review must be in the range of 1-5. if the text is in between positive and negative then return positive. Also  based on the text  tell   will the user  return  or  not likely to return . don't halusinate or guess any output.  "),
    ("user", "text : {text}")
])
    
structured_model = model.with_structured_output(Review)


review = "product are good but service is bad. the person who helped me was rude but the product was good and the food was delicious but need little more hygiene"

prompt = template.invoke({
    "text":review,
})


response:Optional[Review] = structured_model.invoke(prompt)

print("Review : ",response["review"])
print("Sentiment : ",response["sentiment"])    
print("Likelihood of user returning to shop : ",response["return_likely"])
print("Pros : ",response["pros"])
print("Cons : ",response["cons"])