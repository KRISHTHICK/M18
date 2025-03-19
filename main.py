import faiss
import numpy as np
from transformers import BertModel, BertTokenizer
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample data (replace with your own dataset)
documents = [
    "Hello, how can I help you?",
    "What is the weather like today?",
    "Tell me a joke.",
    "I need assistance with my order.",
    "What is your return policy?"
]

# Encode documents using BERT
def encode_documents(documents):
    inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Create FAISS index
def create_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Encode documents and create index
embeddings = encode_documents(documents)
index = create_index(embeddings)

# Function to search for the most similar document
def search(query, k=1):
    query_embedding = encode_documents([query])
    _, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Define chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_query = request.json.get('query')
    response = search(user_query)
    return jsonify(response=response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
