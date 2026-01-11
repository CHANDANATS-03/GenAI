from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor,tool
import datetime

load_dotenv()

# Step 1: LLM Setup
llm = ChatOpenAI(model="gpt-4")

# Step 2: Define Tools
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


query = "What is the current time in Toronto? (You are in India). Just show the current time and not the date"
prompt_template = hub.pull("hwchase17/react")  # URL - https://smith.langchain.com/hub
tools = [get_system_time]
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": query})


