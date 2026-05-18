from  langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable,RunnableParallel
from langchain_core.output_parsers import StrOutputParser


model1:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=config("GOOGLE_API_KEY"))
model2:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))


prompt1:PromptTemplate = PromptTemplate(
    template=""" Generate  short and simple notes from  the following \n {text}""",
    input_variables = ["text"],
    validate_template=True,
)

prompt2:PromptTemplate = PromptTemplate(
    template=""" Generate 5 short question  from  the  following text \n {text}""",
    input_variables = ["text"],
    validate_template=True,
)


prompt3:PromptTemplate = PromptTemplate(
    template="""" 
    merge the provided notes and quize in a single documents \n
    notes -> {notes} \n 
    quize -> {quize}
    """,
    input_variables = ["notes","quize"],
    validate_template=True,
)



parser:StrOutputParser = StrOutputParser()

prompt1_chain = prompt1 | model1 | parser
prompt2_chain = prompt2 | model2 | parser

peralal_chain = RunnableParallel({
    "notes":prompt1_chain,
    "quize":prompt2_chain,
})

final_chain = peralal_chain | prompt3 | model2 | parser 

text:str = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

"""
response = final_chain.invoke({"text":text})
print(response)

final_chain.get_graph().print_ascii()
