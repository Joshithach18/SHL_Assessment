from flask import Flask, jsonify, request
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load model and data
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
data =pd.read_csv("shlassessments.csv", encoding='utf-8', encoding_errors='ignore')
data['combined_text'] = data.apply(lambda row: " ".join(row.dropna().astype(str)), axis=1)

# Create FAISS index
embeddings = model.encode(data['combined_text'].tolist(), convert_to_tensor=True)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings.numpy())

@app.route('/', methods=['GET'])
def home():
    """Default route for the API."""
    return jsonify({"message": "Welcome to the SHL Assessment Recommendation API"}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """Recommendation endpoint to find the top assessments."""
    content = request.json
    query = content.get("query", "")
    top_n = int(content.get("top_n", 10))

    # Encode the query and search
    query_embedding = model.encode([query], convert_to_tensor=True)
    scores, indices = faiss_index.search(query_embedding.numpy(), top_n)

    # Format results
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