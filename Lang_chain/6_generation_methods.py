from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import json

load_dotenv() # load environment variables from .env file

class Movie(BaseModel):
    title: str = Field( description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The release year of the movie")

# initialize the OpenAI chat model with output formatting
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0).with_structured_output(Movie)

# Define a prompt template
template = PromptTemplate.from_template(
    """You are a helpful assistant that answers the details about the movie which the user asks.
    Context: {context}
    Question: {question}
    Answer: """
)

user_input = {
    "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010. It is a science fiction film that explores the concept of shared dreams and the manipulation of the subconscious.",
    "question": "What is the title, director, and year of release of the movie?"
}

chain = template | model  # create a chain that combines the template and the models
# single input and single output
response = chain.invoke(user_input)
print(response)  # print the response from the model
print("--------------------------------")


# Convert the output to json 
response_json = response.model_dump_json(indent=2)
print("Formatted JSON Output:")
print(response_json)  # print the json formatted response

# Batch Demo
batch_user_inputs = [{
    "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010. It is a science fiction film that explores the concept of shared dreams and the manipulation of the subconscious.",
    "question": "What is the title, director, and year of release of the movie?"
},{
    "context": "The movie 'The Matrix' is directed by Lana Wachowski and Lilly Wachowski and was released in 1999. It is a science fiction film that explores the nature of reality and human existence.",
    "question": "What is the title, director, and year of release of the movie?"
} ,
{
    "context": "The movie 'The Shawshank Redemption' is directed by Frank Darabont and was released in 1994. It is a drama film that tells the story",
    "question": "What is the title, director, and year of release of the movie?"
}]
print("--------------------------------")
# Batch processing
batch_response = chain.batch(batch_user_inputs)
print("Batch Response:")
for i, res in enumerate(batch_response):
    print(f"Response {i+1}:")
    print(res)  # print the response from the model
    print("--------------------------------")
    # Convert the output to json 
    response_json = res.model_dump_json(indent=2)
    print("Formatted JSON Output:")
    print(response_json)  # print the json formatted response

# Streaming demo
print("Streaming Response:")
print("--------------------------------")
for token in chain.stream(user_input):
    print(token, end="", flush=True)  # print the token as it is generated