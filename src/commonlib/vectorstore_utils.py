from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

logger = logging.getLogger(__name__)

# Global storage for multiple vectorstores
_vectorstores: Dict[str, InMemoryVectorStore] = {}

def get_vectorstore(thread_id: str):
    global _vectorstores
    
    if thread_id not in _vectorstores:
        logger.info(f"Initializing vectorstore for thread: {thread_id}")
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        _vectorstores[thread_id] = InMemoryVectorStore(_embeddings)
        logger.info(f"Created new vectorstore for thread: {thread_id}")
    
    logger.info(f"Returning vectorstore for thread: {thread_id}")
    return _vectorstores[thread_id]

def build_vectorstore(thread_id: str, urls: List[str]):
    vectorstore = get_vectorstore(thread_id)
    
    if len(vectorstore.store.items()) == 0:
        logger.info(f"Building vectorstore from URLs: {urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        vectorstore.add_documents(documents=doc_splits)
    else:
        logger.info(f"Vectorstore for thread: {thread_id} already exists.")
    
    retriever = vectorstore.as_retriever()
    logger.info(f"Vectorstore built successfully and retriever created for thread: {thread_id}")
    return retriever