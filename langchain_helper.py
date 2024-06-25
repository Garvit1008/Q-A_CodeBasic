import logging
from langchain_community.chat_models import ChatGooglePalm
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configuration
api_key = "AIzaSyDwrgGw2CW6Mcmwk6iS-_5mX6VPnKkuoeQ"
vectordb_file_path = "faiss_index"
embeddings = HuggingFaceInstructEmbeddings()

def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt", encoding="latin-1")
    data = loader.load()
    vectordb = FAISS.from_documents(data, embeddings)
    vectordb.save_local(vectordb_file_path)
    index_file_path = os.path.join(vectordb_file_path, "index.faiss")
    if not os.path.exists(index_file_path):
        logging.error(f"FAISS index file '{index_file_path}' was not created.")
        raise FileNotFoundError(f"FAISS index file '{index_file_path}' was not created.")
    logging.info(f"FAISS index file '{index_file_path}' successfully created.")

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say thanks.
    context:{context}
    question:{question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=ChatGooglePalm(google_api_key=api_key, temperature=0.3),
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain
if __name__ == '__main__':
    chain = get_qa_chain()
    query = "do you provide internship?"
    response = chain(query)
    print(response)