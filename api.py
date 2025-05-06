from flask import Flask, jsonify, request
import pandas as pd
import faiss
import cohere
import numpy as np

app = Flask(__name__)

# Initialize Cohere client
COHERE_API_KEY = "QU0eJVAl4MbACkDCy9WPN640qiViL1po6Z6kPr8S"  # Replace with your actual key
co = cohere.Client(COHERE_API_KEY)

# Load CSV data
data = pd.read_csv("shlassessments.csv", encoding='utf-8', encoding_errors='ignore')
data['combined_text'] = data.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)

# Generate Cohere embeddings
print("Generating embeddings from Cohere...")
response = co.embed(texts=data['combined_text'].tolist(), model="embed-english-v3.0", input_type="search_document")
embeddings = np.array(response.embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the SHL Assessment Recommendation API"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    content = request.json
    query = content.get("query", "")
    top_n = int(content.get("top_n", 10))

    # Generate embedding for the query using Cohere
    query_response = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query")
    query_embedding = np.array(query_response.embeddings)

    # FAISS search
    scores, indices = faiss_index.search(query_embedding, top_n)

    # Format and return results
    results = []
    for idx in indices[0]:
        result = {
            "Assessment Name": data.iloc[idx]["Name"],
            "URL": data.iloc[idx]["URL"],
            "Remote Testing Support": data.iloc[idx]["Remote testing support"],
            "Adaptive/IRT Support": data.iloc[idx]["Adaptive/IRT support"],
            "Duration": data.iloc[idx]["Duration"],
            "Test Type": data.iloc[idx]["Test type"]
        }
        results.append(result)

    return jsonify(results), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
