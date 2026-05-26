from  langchain_google_genai import GoogleGenerativeAIEmbeddings
from decouple import config
from typing import List


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", api_key=config("GOOGLE_API_KEY") , output_dimension=32)


documents:List[str] = ["Dhaka  is the capital of Bangladesh",
                     "Software Industry is growing fast",
                      "Garments Industry is growing fast",
                      "Bangladesh is a developing country",
                       ]

response = embeddings.embed_documents(documents)
print(str(response))