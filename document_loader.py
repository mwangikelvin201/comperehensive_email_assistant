from helper import load_pdf,splitter, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore 

import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# print(PINECONE_API_KEY)

extracted_data = load_pdf("data/")
text_chunks = splitter(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)



index_name="companypolicy"

#Creating Embeddings for Each of The Text Chunks & storing
vector_store = PineconeVectorStore(index_name=index_name,embedding=embeddings)
doc_search = vector_store.add_documents(documents= text_chunks )
print(f"Added {len(doc_search)} documents to the vector store.")
