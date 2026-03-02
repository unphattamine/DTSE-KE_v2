import os
import chromadb
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import LLMRerank

st.set_page_config(page_title="DTSE RAG", layout="wide")
load_dotenv()
api_key = os.getenv("API_KEY")

st.sidebar.title("Search Settings")
llm_model_options = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "GPT-OSS 120B": "openai/gpt-oss-120b",
    "Kimi K2 Instruct": "moonshotai/kimi-k2-instruct-0905"
}
emb_model_options = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5"
}

selected_emb_model = st.sidebar.selectbox("Embedding model:", options=list(emb_model_options.keys()), index=0)
selected_gen_model = st.sidebar.selectbox("Summarization model:", options=list(llm_model_options.keys()), index=1) #120B
# st.sidebar.subheader("Re-ranker")
selected_rerank_model = st.sidebar.selectbox("Data ranking Model:", options=list(llm_model_options.keys()),index=0) #70B
Settings.embed_model = HuggingFaceEmbedding(model_name=emb_model_options[selected_emb_model])

def check_files():
    if not os.path.exists("./data") or not os.listdir("./data"):
        st.error("No files in /data folder")
        st.stop()

def count_all_files():
    check_files()
    return sum(1 for item in Path("./data").iterdir() if item.is_file())

@st.cache_resource(show_spinner="Updating database...")
def initialize_chat_engine(gen_model_id, rerank_model_id, emb_model_id):
    #llm init
    gen_llm = Groq(model=llm_model_options[gen_model_id], api_key=api_key)
    rerank_llm = Groq(model=llm_model_options[rerank_model_id], api_key=api_key)

    #settings
    Settings.llm = gen_llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=emb_model_options[emb_model_id])
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    #db
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("my_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #data
    check_files()
    documents = SimpleDirectoryReader("./data").load_data()
    nodes = Settings.text_splitter.get_nodes_from_documents(documents)

    #vector index
    if chroma_collection.count() == 0:
        index = VectorStoreIndex(nodes, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

    #retrievers
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)
    vector_retriever = index.as_retriever(similarity_top_k=4)

    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=5,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True,
    )

    #reranker
    reranker = LLMRerank(
        llm=rerank_llm,
        choice_batch_size=5,
        top_n=3
    )

    #prompts
    system_prompt = (
        "You are a Deutsche Telekom press assistant."
        "For greetings or general small talk (like 'hello' or 'tell me something'), respond politely as a professional assistant."
        "For questions about deutsche telekom, answer ONLY based on the provided context. "
        "If you dont have the information, say: 'Sorry, I don't know, but ask me tomorrow.'"
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    chat_engine = ContextChatEngine.from_defaults(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        system_prompt=system_prompt,
        chat_mode="condense_plus_context",
        memory=memory,
        verbose=True
    )

    return chat_engine

st.sidebar.divider()
st.sidebar.info(f"**Data:** {count_all_files()} files in ./data")

if st.sidebar.button("Restart Chat"):
    st.session_state.messages = []
    st.rerun()

chat_engine = initialize_chat_engine(selected_gen_model, selected_rerank_model, selected_emb_model)

st.title("Deutsche Telekom Press release RAG")
if "messages" not in st.session_state:
    st.session_state.messages = []

#chat context
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#input
if prompt := st.chat_input("Ask something about recent press releases"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat_engine.chat(prompt)
        if response.response.lower() == "empty response":
            st.markdown("Sorry, i dont have any information on that")
        else:
            st.markdown(response.response)

        with st.expander("Sources"):
            for i, node in enumerate(response.source_nodes):
                score = getattr(node, 'score', 'N/A')
                st.markdown(f"**Source {i + 1} (Score: {score}):**")
                st.caption(node.node.get_content()[:500] + "...")
                st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response.response})