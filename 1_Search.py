from typing import List
from llama_index.core.schema import Document
from llama_index.llms.ollama import Ollama
import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import (SentenceSplitter, TokenTextSplitter, LangchainNodeParser)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv
import openai
import os
import pandas as pd
import shelve
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.chat_engine import CondenseQuestionChatEngine


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


st.set_page_config(
    page_title="Search",
    page_icon="🔍",
)


# Data directory
data_directory = r"D:\AI_Stuffs\data"

# Custom Excel reader
class ExcelFileReader(BaseReader):
    def load_data(self, file, extra_info=None) -> List[Document]:
        df = pd.read_excel(file)
        text = df.to_string(index=False)
        return [Document(text=text + " Foobar", extra_info=extra_info or {})]

# Read documents from directory
file_extractor = {".xlsx": ExcelFileReader()}
documents = SimpleDirectoryReader(input_dir=data_directory, filename_as_id=True, file_extractor=file_extractor).load_data()

# Set up the LLM and embedding model
Settings.llm = Ollama(model="gemma2", temperature=0.3, request_timeout=1000,  additional_kwargs={"num_predict": 150})
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Response synthesizer
response_synthesizer = get_response_synthesizer(
    response_mode="compact", use_async=True
)

# Sentence splitter
splitter = LangchainNodeParser(RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200))

Settings.transformations = [splitter]

# Vector store
@st.cache_resource
def vectorstore():
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("shopping")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    # Re-initialize ChromaDB client and collection
    db2 = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db2.get_or_create_collection("shopping")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Load index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        transformations=[splitter]
    )
    return index

index = vectorstore()

retriever = VectorIndexRetriever(index = index, similarity_top_k=7)
# Query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# Define QA prompt template
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, follow the steps below and then answer:\n"
    "1. Identify the products that match the criteria given in the question.\n"
    "2. If the context is empty or the answer cannot be determined from the context alone, respond with 'I do not know the answer to the question.'\n"
    "3. Provide only the names of the products that match the criteria.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)


# Update prompts in query engine
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    verbose=True,
)

# #Display
qa_prompt_str = ''' Recommend me 2-3 waterproof mascaras '''

# response = query_engine.query(qa_prompt_str)
# print(response)


#Shelve
# Read the Excel file into a pandas DataFrame
dframe = pd.read_excel('data1.xlsx')

# Select the relevant columns
data = dframe[["Canonical Link", "Title", "About This Item"]]

# Convert the DataFrame to a list of dictionaries
data_dict_list = data.to_dict(orient='records')



#Frontend
st.title('Shopping Assistant')
query = st.text_input("What would you like to ask ?")
# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            response = chat_engine.chat(query)
            st.success(response)
            # Open a shelve database
            with shelve.open('data1') as excel_database_shelve:
            # Store each entry in the shelve database with a unique key
                for idx, row in enumerate(data_dict_list):
                    key = f"row_{idx}"
                    excel_database_shelve[key] = row

            # Function to search for product information by partial title
                def search_by_partial_title(partial_title):
                    results = []
                    for key in excel_database_shelve:
                        product = excel_database_shelve[key]
                        if partial_title.lower() in product["Title"].lower():
                            results.append(product["About This Item"])
                    return results
                partial_title_to_search = str(response)  
                matching_products = search_by_partial_title(partial_title_to_search)

                if matching_products:
                    for product in matching_products:
                        st.write(f"Product Info: {product}")
                else:
                    print(f"No products found with title containing '{partial_title_to_search}'.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

            