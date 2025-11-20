# rss_feed
import asyncio
from typing import TypedDict, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from openai import AsyncOpenAI
from langchain_core.runnables import Runnable
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import FakeEmbeddings
from langgraph.checkpoint.memory import MemorySaver

def dict_to_message(msg_dict):
    role = msg_dict.get("role")
    content = msg_dict.get("content")
    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"Unknown role in message dict: {role}")
    

def create_BM25retriever_from_docs(
    docs: list[Document], 
    k : int
    ):  

    try:  
        if not docs:  
            raise ValueError("The documents list cannot be empty.")  
        if k <= 0:  
            raise ValueError("k must be a positive integer.")  
  
        bm25_retriever = BM25Retriever.from_documents(docs)  
        bm25_retriever.k = k  
        return bm25_retriever  
    except Exception as e:  
        print(f"An error occurred while creating the BM25 retriever: {e}")  
        return None  
    
def create_hybrid_retriever(
    sparse_retriever, 
    semantic_retriever,
     weights_sparse : float
     ):  

    try:  
        if not (0 <= weights_sparse <= 1):  
            raise ValueError("weights_sparse must be between 0 and 1.")  
  
        ensemble_retriever = EnsembleRetriever(  
            retrievers=[sparse_retriever, semantic_retriever],  
            weights=[weights_sparse, 1 - weights_sparse]  
        )  
        return ensemble_retriever  
    except Exception as e:  
        print(f"An error occurred while creating the hybrid retriever: {e}")  
        return None
    

def load_book(path_data, chunk_size: int = 500, chunk_overlap: int = 100):
    text_loader_kwargs = {"autodetect_encoding": True}
    loader = TextLoader(path_data, **text_loader_kwargs)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    clean_docs = splitter.split_documents(docs)
    for doc in clean_docs:
        # Remove all newline characters
        doc.page_content = doc.page_content.replace("\n", " ").strip()
        doc.page_content = doc.page_content.replace("\t", " ").strip()

    for i, doc in enumerate(clean_docs, start=1):
        # Copy existing metadata or create new dict if None
        metadata = dict(doc.metadata) if doc.metadata else {}
        # Add or overwrite the 'id' field with progressive integer
        metadata["id"] = i
        # Update the document's metadata
        doc.metadata = metadata

    return clean_docs
    
def get_hybrid_retrieve(docs, k_docs, vector_store, embeddings_model_name, weights_sparse):
    bm25_retriever = create_BM25retriever_from_docs(docs, k=k_docs)
    retriever = RetrieverRunnable(vector_store=vector_store, default_embedding_model=embeddings_model_name)
    hybrid_retriever = create_hybrid_retriever(bm25_retriever, retriever, weights_sparse=weights_sparse)
    return hybrid_retriever
