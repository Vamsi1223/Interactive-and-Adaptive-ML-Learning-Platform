from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Initialize embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# List of resources
resources = [
    "https://www.geeksforgeeks.org/ml-linear-regression/",
    "https://www.geeksforgeeks.org/understanding-logistic-regression/",
    "https://www.geeksforgeeks.org/decision-tree/",
    "https://www.geeksforgeeks.org/random-forest-regression-in-python/",
    "https://www.geeksforgeeks.org/unsupervised-learning/",
    "https://www.geeksforgeeks.org/k-means-clustering-introduction/",
    "https://www.geeksforgeeks.org/hierarchical-clustering/",
    "https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/",
    "https://www.geeksforgeeks.org/ml-mean-shift-clustering/",
    "https://www.geeksforgeeks.org/ml-spectral-clustering/",
]

# Extract and process documents
documents = [extract_text_from_url(url) for url in resources]
documents = [doc for doc in documents if doc]  # Remove None values

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = [chunk for doc in documents for chunk in text_splitter.split_text(doc)]

# Generate embeddings and create FAISS index
vector_db = FAISS.from_texts(chunks, embedding_model)

# Save the FAISS index
vector_db.save_local("ml_resources_index")

print("FAISS index saved successfully!")
