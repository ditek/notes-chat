import streamlit as st

from rag import (
    create_embeddings,
    load_vector_store,
    create_llm,
    answer_question,
)


@st.cache_resource
def get_rag_resources():
    embeddings = create_embeddings()
    vector_store = load_vector_store(embeddings)
    llm = create_llm()
    return vector_store, llm


st.title("Notes Q&A")

with st.sidebar:
    k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)
    if st.button("Reload index"):
        get_rag_resources.clear()
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, doc in enumerate(message["sources"], start=1):
                    source = doc.metadata.get("source", "unknown")
                    chunk_id = doc.metadata.get("chunk_id", "?")

                    st.markdown(f"- [{i}] {source} chunk {chunk_id}")
                    st.expander(f"Content of source {i}").markdown(doc.page_content)

question = st.chat_input("Ask my notes a question")
if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the answer..."):
            vector_store, llm = get_rag_resources()
            answer, docs = answer_question(
                question,
                vector_store,
                llm,
                k=k,
            )
        st.markdown(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                chunk_id = doc.metadata.get("chunk_id", "?")

                st.markdown(f"- [{i}] {source} chunk {chunk_id}")
                st.expander(f"Content of source {i}").markdown(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })
