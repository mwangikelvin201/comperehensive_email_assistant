import os
import openai
import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")
pc = pinecone.Pinecone(api_key=api_key)
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

def initialize_pinecone():
    """Initialize Pinecone vector store."""
    global vectorstore
    try:
        print("🔧 Initializing Pinecone...")
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("⚠️ PINECONE_API_KEY not found")
            return False
            
        pc = pinecone.Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "company-policies")
        
        try:
            indexes = pc.list_indexes()
            if index_name not in [index.name for index in indexes]:
                print(f"⚠️ Index '{index_name}' not found")
                return False
        except Exception as e:
            print(f"⚠️ Could not access Pinecone: {e}")
            return False
            
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print(f"✅ Pinecone connected: {index_name}")
        return True
        
    except Exception as e:
        print(f"⚠️ Pinecone initialization failed: {e}")
        return False

# Initialize services
pinecone_available = initialize_pinecone()

EMAIL_PROMPT_TEMPLATE = """
You are an AI assistant that drafts professional emails based on internal company knowledge.

Using the context below, write a clear and concise email that addresses the user's intent.
Ensure proper grammar, a polite tone, and a logical structure.

CONTEXT:
{context}

USER REQUEST:
{user_query}
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

def query_and_draft_email(user_query: str):
    if not pinecone_available:
        return "Pinecone is not initialized."

    # Step 1: Retrieve relevant documents
    docs = vectorstore.similarity_search(user_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 2: Format prompt
    prompt = PromptTemplate(
        input_variables=["context", "user_query"],
        template=EMAIL_PROMPT_TEMPLATE
    )

    chain = LLMChain(llm=OpenAI(temperature=0.3), prompt=prompt)

    # Step 3: Run chain and return email
    response = chain.run({"context": context, "user_query": user_query})
    return response


if __name__ == "__main__":
    print("📨 Email drafting assistant ready. Type your request or 'exit' to quit.")

    while True:
        user_query = input("\n📝 Enter your email request: ")
        if user_query.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break

        email = query_and_draft_email(user_query)
        print("\n✉️ Generated Email:\n")
        print(email)

