#knowledge_based.py

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader

# Use a lightweight embedding model that works locally
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def create_medical_knowledge_base():
    """Create a simple medical knowledge base"""
    medical_data = [
        {
            "topic": "Common Cold",
            "content": """
            The common cold is a viral infection of the upper respiratory tract. Symptoms typically include
            runny nose, sore throat, cough, congestion, sneezing, low-grade fever, and general malaise.
            Most colds are caused by rhinoviruses. Treatment focuses on relieving symptoms while the body
            fights the infection. Rest, staying hydrated, and over-the-counter medications can help.
            """
        },
        {
            "topic": "Diabetes",
            "content": """
            Diabetes is a chronic condition that affects how the body processes blood sugar (glucose).
            There are two main types: Type 1 (the body doesn't produce insulin) and Type 2 (insulin resistance).
            Symptoms may include increased thirst, frequent urination, hunger, fatigue, and blurred vision.
            Management typically involves monitoring blood sugar, medication, healthy eating, and regular exercise.
            """
        },
        {
            "topic": "Hypertension",
            "content": """
            Hypertension, or high blood pressure, is a condition where the force of blood against artery walls
            is consistently too high. It often has no symptoms but can lead to heart disease and stroke if untreated.
            Blood pressure is measured in two numbers: systolic (top number) and diastolic (bottom number).
            Normal blood pressure is below 120/80 mm Hg. Treatment may include lifestyle changes and medication.
            """
        },
        {
            "topic": "Asthma",
            "content": """
            Asthma is a chronic condition affecting the airways in the lungs. It causes inflammation and narrowing
            of the airways, leading to wheezing, shortness of breath, chest tightness, and coughing.
            Triggers can include allergens, exercise, respiratory infections, and air pollutants.
            Treatment typically involves avoiding triggers and using medications like bronchodilators and inhaled corticosteroids.
            """
        },
        {
            "topic": "Migraine",
            "content": """
            Migraines are recurring headaches that cause moderate to severe throbbing or pulsing pain,
            typically on one side of the head. Symptoms often include nausea, vomiting, and sensitivity to light and sound.
            Some people experience an aura before the headache, which can include visual disturbances.
            Treatment options include pain relievers, preventive medications, and lifestyle management.
            """
        }
    ]
    
    # Load data into LangChain
    df = pd.DataFrame(medical_data)
    loader = DataFrameLoader(df, page_content_column="content")
    documents = loader.load()
    
    # Add metadata
    for i, doc in enumerate(documents):
        doc.metadata["topic"] = medical_data[i]["topic"]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # Save vector store
    vector_store.save_local("medical_knowledge_base")
    
    print(f"✅ Created medical knowledge base with {len(split_docs)} chunks.")
    return vector_store

def load_or_create_knowledge_base():
    """Load or create a medical knowledge base"""
    try:
        vector_store = FAISS.load_local("medical_knowledge_base", embeddings)
        print("✅ Loaded existing medical knowledge base.")
    except:
        print("⚠️ Knowledge base not found. Creating new one...")
        vector_store = create_medical_knowledge_base()
    
    return vector_store

if __name__ == "__main__":
    kb = load_or_create_knowledge_base()

    # Test query
    query = "What are the symptoms of diabetes?"
    docs = kb.similarity_search(query, k=2)

    print("\n🔎 Query:", query)
    print("\n📄 Retrieved documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Topic: {doc.metadata.get('topic', 'Unknown')}")
        print(f"Content: {doc.page_content}")
