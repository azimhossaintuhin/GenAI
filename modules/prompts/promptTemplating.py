from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from decouple import config
import os
from typing import Dict, List



model =  ChatGoogleGenerativeAI(model="gemini-3-flash-preview", api_key=config("GOOGLE_API_KEY"))



optings:Dict[str, List[str]] = {
    "paper inputs":["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"],
    "style inputs":["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
    "length inputs": ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"],
}

template = PromptTemplate(
    template = """Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:
   - Include relevant mathematical equations if present in the paper.
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
   - Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
 input_variables = ["paper_input", "style_input", "length_input"],
 validate_template=True,

)

template.save("template.json")
for paper_input in optings["paper inputs"]:
    print(f"Paper: {paper_input}")

for style_input in optings["style inputs"]:
    print(f"Style: {style_input}")
    
for length_input in optings["length inputs"]:
    print(f"Length: {length_input}")
    
paper = int(input("Enter the paper input: "))
style = int(input("Enter the style input: "))
length = int(input("Enter the length input: "))

prompt = template.format(paper_input=optings["paper inputs"][paper-1], style_input=optings["style inputs"][style-1], length_input=optings["length inputs"][length-1])


response = model.invoke(prompt)
print(response.content[0].get("text","No response"))