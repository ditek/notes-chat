import streamlit as st

from rag import (
    create_embeddings,
    load_vector_store,
    create_llm,
    answer_question,
)

st.title("Notes Q&A")

question = st.text_input("Ask my notes a question")
if question:
    with st.spinner("Finding the answer..."):
        embeddings = create_embeddings()
        vector_store = load_vector_store(embeddings)
        llm = create_llm()
        answer, docs = answer_question(
            question,
            vector_store,
            llm,
            k=3,
        )
    st.markdown(f"**Answer:** {answer}")
    with st.expander("Sources"):
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "?")
            start = doc.metadata.get("start", "?")
            end = doc.metadata.get("end", "?")
            st.markdown(f"- [{i}] {source} chunk {chunk_id} chars {start}-{end}")
            st.expander(f"Content of source {i}").markdown(f"```{doc.page_content}```")