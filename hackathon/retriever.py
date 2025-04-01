import faiss
import json
import requests
from bs4 import BeautifulSoup
import pdfplumber
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Provided resources
resources = [
    {"type": "web", "title": "Machine Learning by Andrew Ng", "url": "https://www.coursera.org/learn/machine-learning"},
    {"type": "web", "title": "Machine Learning Crash Course by Google", "url": "https://developers.google.com/machine-learning/crash-course"},
    {"type": "pdf", "title": "A Few Useful Things to Know About Machine Learning", "path": "https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf"}
]

# Function to extract text from web pages
def extract_web_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return ""

# Function to extract text from PDFs
def extract_pdf_content(pdf_path):
    try:
        if pdf_path.startswith("http"):
            response = requests.get(pdf_path)
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            pdf_path = "temp.pdf"

        text = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text.append(page.get_text("text"))
        return " ".join(text)
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
        return ""

# Extract text from resources
documents = []
for res in resources:
    if res["type"] == "web":
        content = extract_web_content(res["url"])
    elif res["type"] == "pdf":
        content = extract_pdf_content(res["path"])
    else:
        content = ""
    
    if content:
        documents.append({"title": res["title"], "text": content})

# Convert text to embeddings
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, convert_to_numpy=True)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_resources.index")

# Save metadata
with open("metadata.json", "w") as f:
    json.dump(documents, f)

print("FAISS index created and saved!")

# Function to search in FAISS
def search_faiss(query, top_k=3):
    index = faiss.read_index("faiss_resources.index")
    with open("metadata.json", "r") as f:
        documents = json.load(f)

    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(documents):
            results.append(documents[idx]["title"])
    
    return results

# Example Query
query = "Neural networks fundamentals"
similar_docs = search_faiss(query)
print(f"Top related documents: {similar_docs}")
