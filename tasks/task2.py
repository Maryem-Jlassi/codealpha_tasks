import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from PyPDF2 import PdfReader

CSV_PATH = "data/combined_dataset.csv"
DOCS_DIR = "static/documents"  
MODEL = "llama3.2:latest"  

embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  

def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_all_documents():
    all_chunks = []
    
    try:
        df = pd.read_csv(CSV_PATH, delimiter=",", encoding="utf-8")
        for i, row in df.iterrows():
            question = row.iloc[0]  
            answer = row.iloc[1]    
            all_chunks.append(f"Question: {question} Answer: {answer}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
    
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    print(f"PDF document loaded: {filename}, {len(text_chunks)} chunks extracted")
                
                elif filename.lower().endswith('.txt'):
                    text = extract_text_from_txt(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    print(f"TXT document loaded: {filename}, {len(text_chunks)} chunks extracted")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    else:
        print(f"The directory {DOCS_DIR} does not exist. Please create it and add your documents.")
    
    return all_chunks

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, k=10):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
    
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

def build_rag_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": """You are a knowledgeable assistant specialized in providing support for refugees. 
Respond clearly and concisely, using  the informations provided in the context. 
If the context does not contain enough information, ask for clarification or additional details, but rely on the available data. 
Avoid making up information. Be professional and empathetic in your responses."""},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
    }

def main():
    print("Loading the refugee knowledge base...")
    
    all_chunks = load_all_documents()
    
    if not all_chunks:
        print("No data found. Please check your source files.")
        return
    
    print(f"Knowledge base loaded successfully! {len(all_chunks)} text segments available.")
    
    index, chunks = create_faiss_index(all_chunks)
    
    print("Refugee Support Chatbot - Ask your questions (type 'exit' to quit)")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
        
        prompt = build_rag_prompt(query, relevant_chunks)
        
        try:
            response = ollama.chat(**prompt)
            print("\nAI Response:", response['message']['content'])
        except Exception as e:
            print(f"Error with Ollama: {e}")

if __name__ == "__main__":
    main()