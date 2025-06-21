from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
import logging


logger = logging.getLogger(__name__)


def build_vectorstore(thread_id: str, urls: List[str]):
    persist_directory = f"./chroma_db"
    embd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    collection_name = f"{thread_id}"

    # Check if collection exists using ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)
    existing_collections = [c.name for c in client.list_collections()]
    
    if collection_name in existing_collections:
        logger.info("Database directory exists and is not empty")
        # Load existing database
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embd,
            collection_name=collection_name,
        )
    else:
        logger.info(f"Building vectorstore from URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=collection_name,
            embedding=embd,
            persist_directory=persist_directory,
        )
        
    retriever = vectorstore.as_retriever()
    logger.info("Vectorstore built successfully and retriever created.")
    return retriever