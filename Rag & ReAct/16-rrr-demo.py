from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_core.runnables import chain
# Load environment variables

load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

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
    chunk_size=1000,  # Adjust chunk size as needed
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
    embedding=embeddings,
    connection=connection,
    use_jsonb=True,  # Use JSONB for metadata storage
)

# create retriever

retriever = vector_store.as_retriever(search_kwargs= {"k":2})
query = "What is the exp in genrativ AI of Arjun Varma?"
# query = "What is the experience in generative AI?"
rewrite_query = ChatPromptTemplate.from_template(
    """
    Rewrite the better search query to answer the given question related to the context of Generative AI Resume.
    Question: {query}
    Answer:
    """
)

prompt = ChatPromptTemplate.from_template(
    """"Answer the question based only on the provided context.
    context: {context}
    question: {question}"""
)

def parse_rewrite_query(message):
    return message.content.strip() # you can specify any other output parser

rewrite_chain = rewrite_query | llm | parse_rewrite_query


@chain
def rrr_rag_pipeline(query):
    new_query = rewrite_chain.invoke(query)
    print(f"New query: {new_query}")
    docs = retriever.invoke(new_query)
    llm_chain = prompt | llm
    user_input = {"context": docs,
              "question":new_query }
    result = llm_chain.invoke(user_input)
    return result.content

result = rrr_rag_pipeline.invoke(query)
print("Output from RRR RAG:")
print(result)

