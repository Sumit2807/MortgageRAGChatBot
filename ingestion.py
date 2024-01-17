import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
import pinecone
from dotenv import load_dotenv

from consts import INDEX_NAME

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)

def ingest_docs() -> None:
    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    # raw_documents = loader.load()
    # print(f"loaded {len(raw_documents) }documents")
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    # )
    # documents = text_splitter.split_documents(documents=raw_documents)
    # print(f"Splitted into {len(documents)} chunks")

    # for doc in documents:
    #     old_path = doc.metadata["source"]
    #     new_url = old_path.replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})
    loader = PyPDFLoader("/Users/sugupta/RAG_MortgageAssistant/MortgagePolicy/MortgageLendingPolicy.pdf")
    pages = loader.load_and_split()
    # print(f"Page Count: {len(pages)}")
    # print(f"Page Count: {pages.count('Policy')}")

    print(f"Going to insert {len(pages)} to Pinecone")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(pages, embeddings, index_name=INDEX_NAME)
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
