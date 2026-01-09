from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

sys_prompt_template=PromptTemplate.from_template("""answer the following questions using the give context. if the context does not provoide enough information say 'i don't know'. context: {context}   question: {question}""")

sys_prompt= sys_prompt_template.invoke({
    "context": "The capital of india is New Delhi",
    "question": "what is the capital of France?"
})
print("system prompt: ")
print(sys_prompt)
print("---------------------------------")

response=model.invoke(sys_prompt)
print(response.content)

