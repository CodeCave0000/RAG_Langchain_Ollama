
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("C:\\Users\\ranya\\OneDrive\\Desktop\\SamsungGalaxyTabS3.pdf")
loaded_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
document_chunks = splitter.split_documents(loaded_docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(document_chunks, embeddings)
retriever = vector_store.as_retriever(k=3)

query = "What was purchased in this receipt?"
relevant_docs = retriever.invoke(query)
context = "\n\n".join(d.page_content for d in relevant_docs)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a cynical consultant"),
        ("human","Context:{context}\n\nQuestion:{question}")
    ]
)

llm = ChatOllama(model="llama3.1:8b")
response = llm.invoke(prompt_template.format(context=context, question=query))
print(response.content)


