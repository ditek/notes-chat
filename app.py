import os

import streamlit as st

from rag import (
    DEFAULT_NOTES_DIR,
    index_notes,
    sync_notes_from_github,
    create_embeddings,
    load_vector_store,
    create_llm,
    answer_question,
)

AVATARS = {
    "user": "👤",
    "assistant": "📘",
}

@st.cache_resource
def get_rag_resources():
    embeddings = create_embeddings()
    vector_store = load_vector_store(embeddings)
    llm = create_llm()
    return vector_store, llm


st.title("Notes Q&A")

enable_admin_controls = os.getenv("ENABLE_ADMIN_CONTROLS", "false").lower() == "true"

with st.sidebar:
    k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)
    if st.button("Reload index"):
        get_rag_resources.clear()
    if st.button("Clear chat"):
        st.session_state.messages = []
    if enable_admin_controls:
        st.divider()
        st.caption(f"Notes source: `{DEFAULT_NOTES_DIR}`")
        if st.button("Sync notes from GitHub"):
            with st.spinner("Downloading notes..."):
                count = sync_notes_from_github(DEFAULT_NOTES_DIR)
            st.success(f"Downloaded {count} notes")
        if st.button("Rebuild index"):
            with st.spinner("Rebuilding index..."):
                index_notes(DEFAULT_NOTES_DIR, reset=True)
                get_rag_resources.clear()
            st.success("Index rebuilt")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"], None)):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, doc in enumerate(message["sources"], start=1):
                    source = doc.metadata.get("source", "unknown")
                    chunk_id = doc.metadata.get("chunk_id", "?")

                    st.markdown(f"- [{i}] {source} chunk {chunk_id}")
                    with st.expander("Contents"):
                        if message.get("retrieval_query"):
                            st.text(f"Search query: {message.get('retrieval_query')}")
                        st.markdown(doc.page_content)

question = st.chat_input("Ask my notes a question")
if question:
    chat_history = st.session_state.messages.copy()
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Finding the answer..."):
            vector_store, llm = get_rag_resources()
            answer, docs, retrieval_query = answer_question(
                question,
                vector_store,
                llm,
                k=k,
                chat_history=chat_history,
            )
        st.markdown(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                chunk_id = doc.metadata.get("chunk_id", "?")

                st.markdown(f"- [{i}] {source} chunk {chunk_id}")
                with st.expander("Contents"):
                    st.text(f"Search query: {retrieval_query}")
                    st.markdown(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
        "retrieval_query": retrieval_query,
    })
