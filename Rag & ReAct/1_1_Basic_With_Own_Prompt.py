from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, tool
import datetime
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Step 1: LLM Setup
llm = ChatOpenAI(model="gpt-4")

# Step 2: Define Tool
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current system time (India local time) in the given format."""
    return datetime.datetime.now().strftime(format)

# Step 3: Custom ReAct Prompt
# Custom ReAct-style prompt (with required variables: tools + tool_names)
custom_prompt = PromptTemplate.from_template(
    """You are an intelligent AI agent that can solve tasks using reasoning and the available tools.  

Follow this process:
- First, understand the question.
- If useful, decide which tool(s) to call.
- Show your reasoning step by step.
- Finally, return the best possible final answer clearly.

Available tools:
{tools}

You can only use these tool names: {tool_names}

When using tools, follow this format exactly:
Thought: Why you need a tool
Action: the tool name
Action Input: the input to the tool
Observation: the result

Repeat as needed until you can give the final answer.

Begin!

Question: {input}
{agent_scratchpad}"""
)

# Step 4: Build Agent
tools = [get_system_time]
agent = create_react_agent(llm, tools, custom_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# verbose=True is used to print the intermediate steps of the agent's reasoning and actions.
# handle_parsing_errors is used to handle any parsing errors gracefully.
# It will print the error and continue the execution instead of stopping.

# Step 5: Run Query
query = "What is the current time in London? (You are in India). Just show the current time and not the date"
agent_executor.invoke({"input": query})
