# SHL Generative AI Intern Assignment – Semantic Assessment Recommender

This project is a **semantic search-based assessment recommendation system** built for SHL's Generative AI Intern Role. It uses **Cohere’s multilingual embeddings** and **FAISS** for efficient vector similarity search and provides relevant SHL assessments based on a natural language query.

## 🔍 Problem Statement

Given a user's query (e.g., "We need to assess problem-solving skills in software engineers"), return the top N SHL assessments that best match the intent, skillset, and domain described.

## 🚀 Features

- 🔎 Semantic similarity search with Cohere embeddings
- ⚡ Fast vector retrieval using FAISS
- 🌐 Flask API for recommendations
- 🧠 Streamlit frontend with real-time query input and recommendations
- 💬 Placeholder for generative explanation logic (e.g., why this match?)
- 📁 Clean modular design for both backend and frontend

---

## 🛠️ Tech Stack

| Component   | Technology Used           |
|-------------|---------------------------|
| Embeddings  | [Cohere](https://cohere.ai) (`embed-english-v3.0`) |
| Vector DB   | [FAISS](https://github.com/facebookresearch/faiss) |
| Backend API | Flask                     |
| Frontend    | Streamlit                 |
| Language    | Python 3.9+               |

---

## 🗂️ Repository Structure

.
├── api.py # Flask backend API

├── app.py # Streamlit frontend app

├── shlassessments.csv # SHL assessment dataset

├── requirements.txt # Python dependencies

└── README.md # This file


---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/oshithach18/SHL_Assessment.git
cd shl-SHL_Assessment
```
###2. Create and Activate a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
###3. Install Dependencies
```bash
pip install -r requirements.txt
```
###4. Set Your Cohere API Key
Create a .env file or export it directly in your shell:

```bash
export COHERE_API_KEY=your_api_key_here
```
Or on Windows:
md
```bash
set COHERE_API_KEY=your_api_key_here
```
###🧪 Running the Project
🖥️ Option 1: Run the Streamlit Frontend
```bash
streamlit run app.py
```
🌐 Option 2: Start the Flask API (Backend only)
```bash
python api.py
```
Send POST requests to http://localhost:5000/recommend with JSON:

```json
{
  "query": "Looking for problem-solving skills assessment",
  "top_n": 3
}
```

###📌 To-Do / Future Enhancements
 Add generative explanations using Cohere or OpenAI

 Save and reload FAISS index for faster boot time

 Containerize the app using Docker

 Deploy to Hugging Face Spaces or Render

###📄 License
This project is open source and available under the MIT License.

🙋‍♀️ Author

Joshitha Chennamsetty

Gen AI Intern Candidate | Passionate about NLP, LLMs, and applied AI

📧 joshithachennamsetty@example.com

🔗 LinkedIn- Joshitha Chennamsetty| GitHub- Joshithach18
---
