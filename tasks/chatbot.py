import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
import os
import logging
from PyPDF2 import PdfReader
from pathlib import Path
import pickle
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
logger = logging.getLogger(__name__)

CSV_PATH = "data/combined_dataset.csv"
DOCS_DIR = "static/documents"
MODEL = "llama3.2:latest"  
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a plain text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(txt_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file {txt_path}: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error reading text file {txt_path}: {e}")
        return ""

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for better context retrieval."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def load_knowledge_base():
    """Load and prepare the knowledge base from CSV and document files."""
    if os.path.exists("faiss_index.index") and os.path.exists("chunks.pkl"):
        index = faiss.read_index("faiss_index.index")
        with open("chunks.pkl", "rb") as f:
            all_chunks = pickle.load(f)
        logger.info("Loaded FAISS index and chunks from cache.")
        return index, all_chunks
    all_chunks = []
    
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
            if len(df.columns) >= 2: 
                for _, row in df.iterrows():
                    question = row.iloc[0]  
                    answer = row.iloc[1]
                    all_chunks.append(f"Question: {question} Answer: {answer}")
            logger.info(f"Loaded {len(df)} QA pairs from CSV")
        else:
            logger.warning(f"CSV file not found: {CSV_PATH}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
    
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    logger.info(f"Loaded PDF: {filename}, {len(text_chunks)} chunks extracted")
                
                elif filename.lower().endswith('.txt'):
                    text = extract_text_from_txt(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    logger.info(f"Loaded TXT: {filename}, {len(text_chunks)} chunks extracted")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
    else:
        logger.warning(f"Documents directory not found: {DOCS_DIR}")
        Path(DOCS_DIR).mkdir(exist_ok=True, parents=True)
    
    if all_chunks:
        embeddings = embedder.encode(all_chunks, convert_to_tensor=False)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype='float32'))
        logger.info(f"Created FAISS index with {len(all_chunks)} chunks")
        faiss.write_index(index, "faiss_index.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)
        logger.info("FAISS index and chunks saved to disk.")
        return index, all_chunks
    else:
        sample_embedding = embedder.encode(["Placeholder text"], convert_to_tensor=False)
        dimension = sample_embedding.shape[1]
        index = faiss.IndexFlatL2(dimension)
        logger.warning("Created empty FAISS index - no knowledge base content found")
        return index, ["No information available. Please add documents to the knowledge base."]

def retrieve_context(query, index, chunks, k=5):
    """Retrieve the most relevant chunks for a given query."""
    try:
        query_embedding = embedder.encode([query], convert_to_tensor=False)
        distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
        
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(chunks):
                relevant_chunks.append(chunks[idx])
        
        return relevant_chunks
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ["Unable to retrieve relevant information."]
    

def generate_response(query, context):
    """Generate a response using a language model with the retrieved context."""
    try:
        context_text = "\n\n".join(context)
        
        messages = [
            {"role": "system", "content": """You are a knowledgeable assistant specializing in refugee support. 
            Respond clearly and concisely, focusing on the user's query. 
            Avoid acknowledging repetitions or stating that the question has been asked before.
            If the context does not contain enough information, acknowledge the limitations and provide general guidance.
            Be empathetic, supportive, and focus on practical solutions. Use simple language that's easy to understand.
            Always maintain a respectful and helpful tone.
             Respond in a concise (2 or 4 sentences) but conversational way"""},
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}"}
        ]
        
        try:
            response = ollama.chat(model=MODEL, messages=messages)
            answer = response['message']['content']
            return answer
        except Exception as e:
            logger.error(f"Error with language model: {e}")
            return "I'm sorry, I'm having trouble generating a response right now. Please try again later."
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I encountered an error while processing your question."