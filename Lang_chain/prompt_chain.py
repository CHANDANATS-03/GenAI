from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

template=PromptTemplate.from_template("""answer the following questions using the give context. if the context does not provoide enough information say 'i don't know'. context: {context}   question: {question}""")

user_input={
    "context": "The capital of india is New Delhi. It is the seat of the government",
    "question": "what is the capital of France?"
}
chain= template | model

response=chain.invoke(user_input)
print(response.content)
