from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, time, re
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import JsonOutputParser
from aixplain.factories import ModelFactory

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

json_parser = JsonOutputParser()
selected_model = ModelFactory.get("6646261c6eb563165658bbb1")

MIN_CHARACTER_COUNT = 30
def is_valid_address(text: str) -> bool:
    return not (len(text) < MIN_CHARACTER_COUNT and not re.search(r'[,\d]', text))

def load_vector_store():
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=base_url)
    persist_directory = os.path.join(os.getcwd(), "chroma_db_all")
    file_path = os.path.join(Path(__file__).resolve().parent, "output.txt")
    if not os.path.exists(persist_directory):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator="\n")
        docs = text_splitter.split_documents(documents)
        db_local = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
    else:
        db_local = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    return db_local

db = load_vector_store()

class AddressRequest(BaseModel):
    address: str

@app.post("/parse-address")
def parse_address(request: AddressRequest):
    start_time = time.time()
    query = request.address
    
    if not is_valid_address(query):
        raise HTTPException(status_code=400, detail="Invalid address provided.")
    
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.3},
    )
    relevant_docs = retriever.invoke(query)
    
    combined_input = (
        "Find me the region name and emirate name in the address with the help of the documents provided: "
        + query +
        "\n\nRelevant Documents:\n" +
        "\n\n".join([doc.page_content for doc in relevant_docs]) +
        "\n\nPlease provide an answer based only on the provided documents. Give me either both, region name and the emirate or if one is found, give me that and return Null for the other. The name need not match exactly. If there it looks similar go for it. If the address has part of the name in the document, go for it. Like 'Musaffah' instead of 'Al Musafah'.  Give me a dictionary in json format response. Also add corresponding region code and emirate code if available. If not available, return Null. If you are not sure, return Null. The keys should be 'region_name', 'region_code', 'emirate_name', 'emirate_code'. Only return the valid JSON. NO PREAMBLE"
    )
    
    combined_input_2 = (
        "Find me the addressee name, phone number or/and email, any instruction for the dilivery, villa number or flat number, PO Box number or code, building name or apartment name, street or/and landmark from the address: "
        + query +
        "\n\nReturn a dictionary in json with keys: addressee name (if available), phone number (if available), email (if available), delivery instructions (if available), villa number or flat number (if available), PO Box number or code (if available), floor number, building name or apartment name (if available) and street (if available), landmark (if available). If any of the information is not available, please return Null. Just give me the information without any preface. And return Null if you don't know. Only return the valid JSON. NO PREAMBLE"
    )
    
    result_1 = selected_model.run({'text': combined_input})
    result_2 = selected_model.run({'text': combined_input_2})
    
    try:
        parsed_1 = json_parser.parse(result_1.get('data', ''))
    except Exception as e:
        parsed_1 = {}
    
    try:
        parsed_2 = json_parser.parse(result_2.get('data', ''))
    except Exception as e:
        parsed_2 = {}
    
    final_dict = {**parsed_2, **parsed_1}
    
    end_time = time.time()
    processing_time = end_time - start_time
    final_dict["processing_time"] = f"{processing_time:.4f} seconds"
    
    return final_dict
