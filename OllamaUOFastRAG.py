from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import chromadb

# Uncomment the below line to show additional debugging for the Vector similarity search
#import langchain
#langchain.debug = True

from UOFast.app import UOFastDataArray
from UOFast.app.UOLoader import *

# Function to get Data from UOFast API
#
def get_uo_docs(file_name, dict_fields):
    
    uo_api = 'http://127.0.0.1:8200/UODFile'
    mObject =  UOFastDataArray.file_dict_obj(file_name=file_name, dict_fields=dict_fields)
    loader =   UOLoader(api_path=uo_api,file_obj=mObject)

    return loader

def get_embedding():
    return OllamaEmbeddings(model='nomic-embed-text')

def get_local_llm():
    return Ollama(model="mistral")
    
def get_coll_name():
    return "ollama_u2"

def get_persist_dbname():
    return "./tmp/ChromaNomics"

def clear_vectorstore():
    # loaded to a production environment.   
    client = chromadb.PersistentClient(path=get_persist_dbname())
    try:
        coll = client.get_collection(get_coll_name())
        client.delete_collection(get_coll_name())
    except ValueError as v:
        pass

def perform_uodata_vector_store(file_name, dict_fields,  persist_dir ):
    print("Clearing VectorStore")
    clear_vectorstore
    
    # Next two lines Get the Unidata file from the UOFast API call.
    docs = get_uo_docs(file_name, dict_fields)
    docs_list = docs.load()

    #split the Data from Unidata into chunks, for easier ingestion of data into the Vector database - CHROMA
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1500, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs_list)
    
    print("Total documents extracted from Unidata " + str(len(doc_splits))) # This can be commented

    #convert the Unidata data-chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name=get_coll_name(),
        embedding=get_embedding(), persist_directory = persist_dir, 
    )
    return vectorstore

def perform_vector_RAG(persist_dir, question, vectorstore ):
    #vectorstore = Chroma(persist_directory=persist_dir, embedding_function=get_embedding, collection_name=get_coll_name())
    # Creating the retriever function to pass output to the LLM, after similarity search
    retriever = vectorstore.as_retriever( 
        search_type="mmr",
        search_kwargs={'k': 10,  'fetch_k': 55} #, 'lambda_mult': 0.25}
    )
    
    #perform the RAG using llm, vector 
    perform_rag_template = """Answer the question only based on the following context:
    {context}
    Question: {question}
    """
    perform_rag_prompt = ChatPromptTemplate.from_template(perform_rag_template)
    perform_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | perform_rag_prompt
        | get_local_llm()
        | StrOutputParser()
    )
    return perform_rag_chain.invoke(question)


if __name__ == "__main__":
    # U2 Unidata DICT fields to get data using UOLoader. 
    dict_fields = [ "COMPANY", "NAME", "ADDRESS", "CITY", "STATE", "ZIP", "COUNTRY", "PHONE"]
    
    persist_db = get_persist_dbname()
    vector_st = perform_uodata_vector_store("CLIENTS", dict_fields=dict_fields, persist_dir=persist_db)

    input_question = " "
    while input_question != "":
        if input_question != "":
            input_question = input("Ask the AI a question - ")

            str = perform_vector_RAG(persist_db, input_question, vectorstore=vector_st)

            for string in str.split('\n'):
                print(string)