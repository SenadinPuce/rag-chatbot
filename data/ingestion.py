import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs():
    # Load documents
    loader = PyPDFLoader(file_path="data\\Building Microservices.pdf")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split documents into {len(documents)} chunks")

    # Embed documents
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
    )
    print("Documents ingested into Pinecone")


if __name__ == "__main__":
    ingest_docs()
