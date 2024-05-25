import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import HuggingFaceHub
import numpy as np
import requests


#Extract Information from PDF file
def get_pdf_text(filename):
    text = ""
    pdf_ = PdfReader(filename)
    for page in pdf_.pages:
        text += page.extract_text()
    return text



# iterate over files in
# that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
  docs = []
  for filename in user_pdf_list:
      docs.append(Document( page_content= get_pdf_text(filename), metadata={"name": f"{filename}" , "unique_id":unique_id } ) )
      docs.append(get_pdf_text(filename))
      
  return docs



#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #  384
    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )
    print("done......2")
    Pinecone.from_documents(docs, embeddings, index_name=pinecone_index_name)
    


#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index


def similar_docs_hf(query, final_docs_list, k):

    HF_KEY = "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"
    
    headers = {"Authorization": f"Bearer {HF_KEY}"}
    API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

    payload = {
        "inputs": {
            "source_sentence": query, # query
            "sentences": final_docs_list
        }
      }
    response = requests.post(API_URL, headers=headers, json=payload)

    score_list = response.json()

    
    pairs = list(zip( score_list , final_docs_list))

    # Sort the pairs in descending order of the first element of each pair
    pairs.sort(key=lambda x: x[0], reverse=True)

    # Unzip the pairs back into two lists
    score_list , final_docs_list = zip(*pairs)
    # sorted_list[:k] ,
    return    score_list , final_docs_list 


#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,unique_id):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


def metadata_filename( document ) : 

   text = document.metadata["name"] 
   pattern = r"name=\'(.*?)\'"

  # Use re.findall() to find all matches
   matches = re.search(pattern, text)

   matches = re.findall(pattern, text)

   return matches
      
   
         

def get_summary_hf(relavant_docs ):

  HF_KEY = "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"
  headers = {"Authorization": f"Bearer {HF_KEY}"}
  API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
  headers = {"Authorization": f"Bearer {HF_KEY}"}
  API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
  payload = {
        "inputs": {
            "inputs":  relavant_docs ,
             "parameters": {"do_sample": False}
        }
      }
    
  response = requests.post(API_URL, headers=headers, json=payload)
  return response.json()

# Helps us get the summary of a document
def get_summary(current_doc):
    # llm = OpenAI(temperature=0)
    llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    # chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary
