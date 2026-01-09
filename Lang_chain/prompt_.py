from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

sys_prompt=SystemMessage(content="You are a helpful assistent that answers questions about current events and general knowledge for the childern in the age group 8 to 12 years. Ensure that your responses are simple, clear, and age-appropriate. IF you don't know the answer , say 'I don't know' insted of making up an answer. always provide accurate and factual information")

human_prompt=HumanMessage(content="what are you doing")
respose = model.invoke([sys_prompt, human_prompt])
print(respose.content)