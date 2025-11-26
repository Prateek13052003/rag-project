import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key")

# -----------------------
# Initialize Groq LLM
# -----------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)


prompt = ChatPromptTemplate.from_template("""
Use ONLY the following context to answer the question.

<context>
{context}
</context>

Question: {input}
""")


# -----------------------
# Vector DB Creation Function
# -----------------------
def create_vector_db():

    # Prevent duplicate creation
    if "vectors" in st.session_state:
        return

    st.write("Loading PDFs...")
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    st.write("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    st.write("Generating embeddings (FREE HuggingFace model)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    st.write("Building FAISS Vector Database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save in session
    st.session_state.vectors = vectorstore
    st.session_state.embeddings = embeddings

    st.success("Vector DB successfully created ✔")


# -----------------------
# Streamlit App UI
# -----------------------
st.title("RAG with Groq Only")

query = st.text_input("Ask something:")

if st.button("Build Vector DB"):
    create_vector_db()

# Prevent errors when DB not built yet
if query:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please click **Build Vector DB** first!")
    else:
        retriever = st.session_state.vectors.as_retriever()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        result = rag_chain.invoke({"input": query})

        st.subheader("📘 Answer")
        st.write(result["answer"])

        with st.expander("📚 Retrieved Context"):
            for d in result["context"]:
                st.write(d.page_content)
                st.write("---")
