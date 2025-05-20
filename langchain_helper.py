# Import required modules
from langchain.llms import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings


load_dotenv()# Load environment variables from .env file

# LLM
llm = ChatGroq(
    model='llama3-8b-8192',
    temperature=0  # Temperature 0 for deterministic outputs
)

# Embedding Model
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en")

# Path for vector database storing/loading
vectordb_file_path = r"LangChain\QA_System\app\faiss_index"


def create_vector_db():
    
    loader = CSVLoader(file_path=r'LangChain\QA_System\faqs.csv', source_column="prompt")
    data = loader.load()

    # Create FAISS vector store from the document embeddings
    vectordb = FAISS.from_documents(documents=data, embedding=embedding_model)

    # Save the vector store to disk
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the saved FAISS vector store
    vectordb = FAISS.load_local(
        vectordb_file_path,             
        embedding_model,
        allow_dangerous_deserialization=True  
    )

    # Create a retriever from the vector store with a similarity score threshold
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define the prompt template used to instruct the LLM
    prompt_template = """in the response only have the result not the peramble Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create the Retrieval-based QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use the "stuff" chain which combines all context into a single string
        retriever=retriever,
        input_key="query",  # Input key expected by the chain
        return_source_documents=True,  # Return source docs for transparency
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


if __name__ == "__main__":
    # Uncomment this line to create the vector DB when needed
    # create_vector_db()

    
    chain = get_qa_chain()# Get the QA chain

    
    print(chain("Do you have javascript course?"))# Ask a sample question
