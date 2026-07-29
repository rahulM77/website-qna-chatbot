import os
import time
from dotenv import load_dotenv
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

# Load all defined environment variables from .env
load_dotenv()

# Streamlit Page Setup
st.set_page_config(page_title="Customer Support QnA Bot", layout="centered")
st.subheader("Customer Support QnA Bot")

# --- INITIALIZE SESSION STATES ---
if "web_loadd" not in st.session_state:
    st.session_state.web_loadd = False

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_urls(urls_list):
    """
    RAG Pipeline Core Processing logic.
    Fixes the bug in video where .append generated invalid nested lists.
    We correctly use .extend to maintain flat lists of loaded documents.
    """
    all_docs = []
    
    # Loop over user-provided URLs
    for url in urls_list:
        clean_url = url.strip()
        if not clean_url:
            continue
        try:
            # Create a header configuration to mimic a standard web browser
            header_config = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            # Pass the spoofed headers directly to WebBaseLoader
            loader = WebBaseLoader(
                web_path=clean_url,
                requests_kwargs={"headers": header_config}
            )
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"Failed to crawl/load data from: {clean_url}. Error: {str(e)}")
            return

    # Text Splitting Engine
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # Building Vector Embedding and storing inside In-Memory Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(
        documents=split_docs, 
        embedding=embeddings
    )

    # Save to Persistent Browser Session State
    st.session_state.vector_db = vector_store
    st.session_state.web_loadd = True


# --- INITIAL LAYOUT VIEW (URL Ingestion Stage) ---
if not st.session_state.web_loadd:
    urls_input = st.text_area(
        label="Enter URLs (One per line)",
        placeholder="https://example.com\nhttps://another-site.com"
    )
    
    # Trigger Processing workflow
    if urls_input:
        lines = urls_input.split("\n")
        filtered_urls = [line.strip() for line in lines if line.strip()]
        
        if filtered_urls:
            with st.spinner("Processing websites..."):
                process_urls(filtered_urls)
            
            if st.session_state.web_loadd:
                st.success("URLs Processed Successfully!")
                time.sleep(2)
                st.rerun()


# --- CHAT VIEW (Active QnA Dialog Engine) ---
if st.session_state.web_loadd and st.session_state.vector_db is not None:
    
    # Render chat messaging history
    for message in st.session_state.messages:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])

    # Collect User Queries
    query = st.chat_input("Ask a question about the websites:")

    if query:
        # Display current human message inside UI
        with st.chat_message("user"):
            st.markdown(query)
            
        # Append message history
        st.session_state.messages.append({"role": "user", "content": query})

        # Similarity context retrieval
        records = st.session_state.vector_db.similarity_search(query=query, k=6)
        
        # Combine retrieved chunks into a context string
        context = ""
        for chunk in records:
            context += chunk.page_content + "\n\n"

        # Model Inference execution (Using stable production model 2.5 Flash)
        llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=st.secrets["GOOGLE_API_KEY"])
        
        prompt = f"""
Give me final answer for my question based on the provided context.

Context:
{context}

Question:
{query}
"""
        try:
            response = llm.invoke(prompt)
            ai_answer = response.content
            
            # Display current Assistant message inside UI
            with st.chat_message("assistant"):
                st.markdown(ai_answer)
                
            # Append reply history
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
        except Exception as e:
            st.error(f"An error occurred while generating response: {str(e)}")
