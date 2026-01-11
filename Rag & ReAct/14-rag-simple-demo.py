from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() # load environment variables from .env file
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

parser = PyPDFParser()

documents = parser.lazy_parse(blob)
docs = []
for doc in documents:
    docs.append(doc)
# print(docs[0].page_content)
# print(docs[0].metadata)
# print(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Adjust chunk size as needed
    chunk_overlap=50  # Adjust overlap as needed
)

chunks = splitter.split_documents(docs)
# print(f"Number of chunks created: {len(chunks)}")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:")
#     print(chunk.page_content)  # Print the content of each chunk
#     print("--------------------------------")

# create vector store
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=model,
    connection=connection,
    use_jsonb=True,  # Use JSONB for metadata storage
)

# # Print the vector store
# print("Vector Store:")
# print(vector_store)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is the experience of Arjun Varma in AI?"

docs = retriever.invoke(query) # retrieve the context based on the query

print("Retrieved documents: ")
for doc in docs:
    print(doc.page_content)
    print("--------------------------------")

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers the details based on the context provided.
    Context: {context}
    Question: {question}
    """
)
 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

chain = prompt | llm

user_input = {"context": docs, "question": query} # pass the context and question to the chain

response = chain.invoke(user_input)

print("Response:")
print(response.content)