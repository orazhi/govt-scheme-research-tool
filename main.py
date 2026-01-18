import requests
import streamlit as st
import os
import pickle
import faiss
import mimetypes
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse

logging.getLogger("unstructured").setLevel(logging.ERROR)

load_dotenv(".config")

PICKLE_FILE = "faiss_store_openai.pkl"
openai_base_url = "" # Enter base url here

llm = ChatOpenAI(base_url=openai_base_url, model="gpt-4.1", temperature=0.3, api_key=os.getenv("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(base_url=openai_base_url, model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Haqdarshak Scheme Research Tool", layout="wide")

def save_vectorstore_to_pkl(vectorstore, filename):
    """Manually serializes the FAISS index and pickles the full state into a SINGLE file to satisfy assignment requirements."""
    index_bytes = faiss.serialize_index(vectorstore.index)
    store_data = {"index_bytes": index_bytes, "docstore": vectorstore.docstore, "index_to_docstore_id": vectorstore.index_to_docstore_id}
    with open(filename, "wb") as f:
        pickle.dump(store_data, f)

def load_vectorstore_from_pkl(filename, embeddings):
    """Restores the vectorstore from the single pickle file."""
    if not os.path.exists(filename): return None
    with open(filename, "rb") as f:
        store_data = pickle.load(f)
    index = faiss.deserialize_index(store_data["index_bytes"])
    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=store_data["docstore"], index_to_docstore_id=store_data["index_to_docstore_id"])
    return vectorstore

def try_unstructured_url_loader(url, headers):
    try:
        loader = UnstructuredURLLoader(urls=[url], headers=headers)
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 0:
            return docs, "Success: UnstructuredURLLoader"
    except Exception as e:
        pass
    return None, None

def try_download_and_load(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').split(';')[0]
        ext = mimetypes.guess_extension(content_type) or ".html"
        if url.endswith(".pdf"): ext = ".pdf"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        docs = None
        method = "Failed"

        try:
            loader = UnstructuredFileLoader(tmp_path, strategy="fast")
            docs = loader.load()
            if docs and len(docs[0].page_content.strip()) > 0:
                method = "Success: UnstructuredFileLoader"
        except Exception:
            pass

        if not docs:
            if ext == ".pdf":
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    method = "Success: PyPDFLoader (Fallback)"
                except: pass
            else:
                try:
                    soup = BeautifulSoup(response.content, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    if len(text) > 100:
                        docs = [Document(page_content=text, metadata={"source": url})]
                        method = "Success: BeautifulSoup (Fallback)"
                except: pass

        os.remove(tmp_path)
        return docs, method

    except Exception as e:
        return None, f"Download Failed: {str(e)}"

def process_urls(urls):
    all_docs = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"}

    with st.status("🚀 initializing Modular Ingestion Pipeline...", expanded=True) as status:
        
        for url in urls:
            status.write(f"🔍 Processing: {url}")
            
            docs, msg = try_unstructured_url_loader(url, headers)
            if docs:
                status.write(f"✅ {msg}")
                all_docs.extend(docs)
                continue

            status.write("⚠️ URL Loader failed. Attempting deep file inspection...")
            docs, msg = try_download_and_load(url, headers)
            
            if docs:
                status.write(f"✅ {msg}")
                all_docs.extend(docs)
            else:
                status.write(f"❌ All methods failed for {url}. ({msg})")

        if not all_docs:
            st.error("🚨 CRITICAL: No data could be extracted from any source.")
            return None, None

        status.write(f"📦 Aggregated {len(all_docs)} documents. Splitting...")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)
        
        if not split_docs:
            st.error("❌ Content found but chunks are empty.")
            return None, None

        status.write(f"🧠 Building FAISS Index with {len(split_docs)} chunks...")
        try:
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            save_vectorstore_to_pkl(vectorstore, PICKLE_FILE)
            status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
            return vectorstore, split_docs
        except Exception as e:
            st.error(f"❌ FAISS Error: {e}")
            return None, None

def generate_summary(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs[:6]])
    prompt = f"""
    You are a Scheme Research Assistant. Read the following text and create a structured summary.
    You MUST cover exactly these four sections with these exact headings:
   
    1. **Scheme Benefits**
    2. **Scheme Application Process**
    3. **Eligibility**
    4. **Documents Required**
   
    If information is missing for a section, state "Information not available in the provided text."
   
    Context:
    {context_text}
    """
    response = llm.invoke(prompt)
    return response.content

st.title("Scheme Research Tool 🔍")
st.markdown("Automated tool to summarize schemes and answer queries.")

with st.sidebar:
    st.header("Data Sources")
    url_input = st.text_area("Enter Scheme Article URLs (one per line):")
    process_btn = st.button("Process URLs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if process_btn and url_input:
    urls_list = [url.strip() for url in url_input.split('\n') if url.strip()]
    if urls_list:
        vectorstore, docs = process_urls(urls_list)
        if vectorstore and docs:
            st.success(f"Data Processed & Saved to {PICKLE_FILE}!")
            st.markdown("### 📝 Scheme Summary")
            with st.spinner("Generating Summary..."):
                summary = generate_summary(docs)
                st.markdown(summary)
    else:
        st.warning("Please enter valid URLs.")

st.divider()
st.header("Ask Questions about the Scheme")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about the scheme...")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        vectorstore = load_vectorstore_from_pkl(PICKLE_FILE, embeddings)
       
        if not vectorstore:
            st.error("Please process a URL first to build the knowledge base.")
        else:
            retriever = vectorstore.as_retriever()
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context.
            If the answer is not in the context, say that you don't know.
           
            <context>
            {context}
            </context>

            Question: {input}
            """)
           
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
           
            response_placeholder = st.empty()
            full_response = ""
            sources = []
           
            for chunk in retrieval_chain.stream({"input": query}):
                if "context" in chunk:
                    sources = chunk["context"]
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    response_placeholder.markdown(full_response + "▌")
           
            response_placeholder.markdown(full_response)
            if sources:
                unique_urls = list({doc.metadata.get("source") for doc in sources})
                st.markdown("---")
                st.markdown("### 🌐 Sources")
                for url in unique_urls:
                    source_snippet = next((doc.page_content[:150] + "..." for doc in sources if doc.metadata.get("source") == url), "")
                    st.write(f"**URL:** {url}")
                    st.caption(f"**Context Summary:** {source_snippet}")
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
