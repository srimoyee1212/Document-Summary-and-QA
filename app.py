import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from sklearn.cluster import KMeans
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
import time

# Step 1: Load the book
def load_book(file_obj, file_extension):
    """Load the content of a book based on its file type."""
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_obj.read())
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load()
            text = "".join(page.page_content for page in pages)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(temp_file.name)
            data = loader.load()
            text = "\n".join(element.page_content for element in data)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        os.remove(temp_file.name)
    text = text.replace('\t', ' ')
    return text

# Step 2: Split the book and embed the text (with FAISS)
def split_and_embed(text, openai_api_key, batch_size=5):
    """Split text into chunks, embed them, and store in FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    vectors = []
    all_docs = []  # Collect documents for adding to FAISS

    # Embed documents in batches to handle rate limits
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_vectors = embeddings.embed_documents([doc.page_content for doc in batch_docs])
        vectors.extend(batch_vectors)
        all_docs.extend(batch_docs)  # Collect the docs for indexing
    
    # Create FAISS vector store
    faiss_index = FAISS.from_documents(all_docs, embeddings)

    return docs, vectors, faiss_index

# Step 3: Cluster embeddings
def cluster_embeddings(vectors, num_clusters):
    num_samples = len(vectors)

    # Ensure that number of clusters does not exceed the number of samples
    if num_samples < num_clusters:
        num_clusters = num_samples

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    closest_indices = [np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_]
    return sorted(closest_indices)

# Step 4: Summarize the chunks
def summarize_chunks(docs, selected_indices, openai_api_key):
    llm3_turbo = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo-16k')
    map_prompt = """
    You are provided with a passage from a document. Your task is to produce a comprehensive summary of this passage. Ensure accuracy and avoid adding any interpretations or extra details not present in the original text. The summary should be at least three paragraphs long and fully capture the essence of the passage.
    ```{text}```
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    selected_docs = [docs[i] for i in selected_indices]
    summary_list = []

    for doc in selected_docs:
        time.sleep(1)  # Sleep to handle rate limits
        chunk_summary = load_summarize_chain(llm=llm3_turbo, chain_type="stuff", prompt=map_prompt_template).run([doc])
        summary_list.append(chunk_summary)

    return "\n".join(summary_list)

# Step 5: Create final summary
def create_final_summary(summaries, openai_api_key):
    llm4 = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=3000, model='gpt-4', request_timeout=120)
    combine_prompt = """
    You are given a series of summarized sections from a document. Your task is to weave these summaries into a single, cohesive, and verbose summary. The reader should be able to understand the main events or points of the document from your summary. Ensure you retain the accuracy of the content and present it in a clear and engaging manner.
    ```{text}```
    COHESIVE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    time.sleep(1)  # Sleep to handle rate limits
    reduce_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=combine_prompt_template)
    final_summary = reduce_chain.run([Document(page_content=summaries)])
    return final_summary

# Step 6: Create QA System
def create_qa_system(faiss_index, openai_api_key):
    qa_llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo')
    qa_chain = RetrievalQA.from_chain_type(llm=qa_llm, chain_type="stuff", retriever=faiss_index.as_retriever())
    return qa_chain

# Main function for the app
def main():
    st.title("Document Summarizer and Q&A")

    uploaded_file = st.file_uploader("Upload a PDF or EPUB file", type=["pdf", "epub"])
    openai_api_key = ""  # Set your OpenAI API key here

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # Button to generate summary
        if st.button("Generate Summary"):
            with st.spinner("Processing..."):
                text = load_book(uploaded_file, file_extension)
                docs, vectors, faiss_index = split_and_embed(text, openai_api_key)
                selected_indices = cluster_embeddings(vectors, 5)  # Use 5 clusters
                summaries = summarize_chunks(docs, selected_indices, openai_api_key)
                final_summary = create_final_summary(summaries, openai_api_key)
                st.subheader("Summary")
                st.write(final_summary)

                # Store the FAISS index for Q&A later
                st.session_state.faiss_index = faiss_index

        # Section for Q&A
        if "faiss_index" in st.session_state:
            qa_chain = create_qa_system(st.session_state.faiss_index, openai_api_key)

            # User input for questions
            question = st.text_input("Ask a question about the document")
            if st.button("Ask"):
                with st.spinner("Finding the answer..."):
                    answer = qa_chain.run(question)
                    st.subheader("Answer")
                    st.write(answer)

if __name__ == "__main__":
    main()
