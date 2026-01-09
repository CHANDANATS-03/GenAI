from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# 1. Define the state schema
class State(TypedDict):
    # "messages" stores the chat history; `add_messages` appends instead of overwriting
    messages: Annotated[list, add_messages]

# 2. Create graph builder
graph_builder = StateGraph(State)

# 3. Initialize LLM (choose your model here)
model = init_chat_model("openai:gpt-5-nano") # 

# 4. Define chatbot node
def chatbot(state: State):
    return {"messages": [model.invoke(state["messages"])]}

# 5. Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 6. Compile graph
graph = graph_builder.compile()

# Save graph as PNG locally
# png_data = graph.get_graph().draw_mermaid_png()
# with open("1_simple_chatbot.png", "wb") as f:
#     f.write(png_data)
# print("✅ Graph saved as 1_simple_chatbot.png")

# 7. Chat loop
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

if __name__ == "__main__":
    print("🤖 Chatbot is ready! Type 'quit', 'exit', or 'q' to stop.\n")
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
