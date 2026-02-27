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

from dotenv import load_dotenv 
import pandas as pd
import numpy as np
import ast

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
    
# --- RetrieverRunnable with async ainvoke ---
class RetrieverRunnable(Runnable):
    def __init__(self, vector_store, default_embedding_model: str = "text-embedding-3-large"):
        self.vector_store = vector_store
        self.default_embedding_model = default_embedding_model
        self.client = AsyncOpenAI()

    def invoke(self, input: str, config: Optional[dict] = None) -> list[Document]:
        # Run the async ainvoke synchronously
        return asyncio.run(self.ainvoke(input, config))
    
    async def ainvoke(self, input: str, config: Optional[dict] = None) -> list[Document]:
        embeddings_model_name = self.default_embedding_model
        k = 10
        if config and "configurable" in config:
            embeddings_model_name = config["configurable"].get("embeddings_model_name", embeddings_model_name)
            k = config["configurable"].get("k", k)
        docs = await self.retrieve_and_format(input, embeddings_model_name, k)
        return docs
    
    async def retrieve_and_format(self, query: str, embeddings_model_name: str, k: int) -> list[Document]:
        query_embedding = await self.get_query_embeddings(query, embeddings_model_name)
        docs = self.vector_store.similarity_search_by_vector(query_embedding, k=k)
        return docs
    
    async def get_query_embeddings(self, query: str, embeddings_model_name: str):
        response = await self.client.embeddings.create(
            input=query,
            model=embeddings_model_name,
            dimensions=1024
        )
        return response.data[0].embedding

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

# --- State TypedDict ---
# --- Define State with messages for memory ---
class State(TypedDict):
    question: str
    context: str
    answer: str
    messages: List[BaseMessage]  # conversation history for memory

# --- Async retrieve node ---
async def retrieve(state: State) -> State:
    docs = await hybrid_retriever.ainvoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}

# --- Async generate node with streaming ---
def make_generate_node(llm: ChatOpenAI):
    async def generate(state: State) -> State:
        # Normalize messages to LangChain message objects
        messages_history = []
        for msg in state.get("messages", []):
            if isinstance(msg, dict):
                messages_history.append(dict_to_message(msg))
            elif isinstance(msg, BaseMessage):
                messages_history.append(msg)
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")

        # Exclude last assistant message to avoid repetition
        if messages_history and isinstance(messages_history[-1], AIMessage):
            messages_for_llm = messages_history[:-1]
        else:
            messages_for_llm = messages_history

        messages = [
            SystemMessage(content="You are a helpful assistant. Use the provided context to answer the question."),
            *messages_for_llm,
            HumanMessage(content=f"Context: {state['context']}\n\nQuestion: {state['question']}"),
        ]

        response_chunks = []
        async for chunk in llm.astream(messages):
            response_chunks.append(chunk.content)
        full_response = "".join(response_chunks)

        # Append current user question and assistant answer to messages for memory
        updated_messages = messages_history + [HumanMessage(content=state["question"]), AIMessage(content=full_response)]

        return {**state, "answer": full_response, "messages": updated_messages}
    return generate


@st.cache_resource
def load_vector_store():
    print("loading vs")
    path_vectorstore = "./vectorstore/book"
    fake_embeddings_model = FakeEmbeddings(size=1536)  # or use your get_embedding_dim function
    vector_store = FAISS.load_local(path_vectorstore, index_name='faiss_index_book', embeddings=fake_embeddings_model, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_vector_store()
# print('loading vs')

@st.cache_data
def load_documents():
    print("loading book")
    return load_book('./data/book.txt')

docs = load_documents()
# print('loading docs')


@st.cache_resource
def get_bm25_retriever(docs, k=10):
    return create_BM25retriever_from_docs(docs, k=k)

@st.cache_resource
def get_retriever_runnable(_vector_store, embeddings_model_name):
    return RetrieverRunnable(vector_store=_vector_store, default_embedding_model=embeddings_model_name)

@st.cache_resource
def get_hybrid_retriever(_bm25_retriever, _retriever_runnable, weights_sparse=0.5):
    return create_hybrid_retriever(_bm25_retriever, _retriever_runnable, weights_sparse=weights_sparse)

@st.cache_resource
def get_generate_node(_llm):
    return make_generate_node(_llm)

@st.cache_resource
def get_graph(_generate_node):
    memory = MemorySaver()  # In-memory checkpointer for memory persistence
    return (
        StateGraph(State)
        .add_node("retrieve", retrieve)
        .add_node("generate", _generate_node)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", END)
        .compile(checkpointer=memory)
    )
# --- Setup ---

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)  # enable streaming
embeddings_model_name = "text-embedding-3-small"
embeddings_model = OpenAIEmbeddings(model=embeddings_model_name)



# print('loading book')

bm25_retriever = get_bm25_retriever(docs, k=10)
retriever_runnable = get_retriever_runnable(vector_store, embeddings_model_name)
hybrid_retriever = get_hybrid_retriever(bm25_retriever, retriever_runnable, weights_sparse=0.5)


generate = get_generate_node(llm)
graph = get_graph(generate)
print('loading graph')

# --- Streamlit UI with async streaming ---

# Helper to convert LangChain message objects to dicts for session state
def message_to_dict(msg):
    if hasattr(msg, "role"):
        return {"role": msg.role, "content": msg.content}
    # Fallback for BaseMessage or unknown types
    return {"role": "assistant" if getattr(msg, "type", "") == "ai" else "user", "content": getattr(msg, "content", "")}

st.title("LangGraph RAG Chatbot with Memory and Streaming")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    # Append user message as dict
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message container with streaming placeholder
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        async def stream_graph():
            response_chunks = []
            thread_id = "default-thread"
            initial_state = {
                "question": prompt,
                "context": "",
                "answer": "",
                "messages": st.session_state.messages,
            }
            async for msg_chunk, metadata in graph.astream(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                if metadata.get("langgraph_node") == "generate" and msg_chunk.content:
                    response_chunks.append(msg_chunk.content)
                    response_placeholder.markdown("".join(response_chunks))
            return "".join(response_chunks)

        # Run streaming and get final answer
        answer = asyncio.run(stream_graph())

    # Append assistant message once after streaming completes
    st.session_state.messages.append({"role": "assistant", "content": answer})