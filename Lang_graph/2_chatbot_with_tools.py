from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch   # Add this import
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition # Add this import
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# 2. Define state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Create graph builder
graph_builder = StateGraph(State)

# 4. Initialize LLM + Tools
model = init_chat_model("openai:gpt-5-nano")
tool = TavilySearch(max_results=2)
tools = [tool]

# Bind tools so the LLM can decide when to call them
llm_with_tools = model.bind_tools(tools)

# 5. Define chatbot node
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# 6. Tool Node (prebuilt)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 7. Add conditional edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")   # loop back after tool use
graph_builder.add_edge(START, "chatbot")

# 8. Compile graph
graph = graph_builder.compile()

# Save graph as PNG locally
# png_data = graph.get_graph().draw_mermaid_png()
# with open("2_chatbot_with_tool.png", "wb") as f:
#     f.write(png_data)
# print("✅ Graph saved as 2_chatbot_with_tool.png")SS

# 9. Chat loop
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    print("🤖 Chatbot with Web Search is ready! Type 'quit', 'exit', or 'q' to stop.\n")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print("Error:", e)
            break