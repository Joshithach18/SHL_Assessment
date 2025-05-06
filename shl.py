import os
import pandas as pd
import numpy as np
import streamlit as st
import cohere
import faiss
from newspaper import Article
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Streamlit page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;
        background-color: #f8f9fa;
    }
    .title-container {
        background-color: #0066cc;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .assessment-card {
        background-color: red;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .assessment-header {
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .assessment-score {
        background-color: green;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .explanation-box {
        background-color: black;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #0066cc;
    }
    .input-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- Set Cohere API Key ---
COHERE_API_KEY = "j7uTsOOCsQS99XqLFRUFWHWzWQzADufa6AuHWxXU"  # Replace with your key
co = cohere.Client(COHERE_API_KEY)

# --- Caching data ---
@st.cache_data
def load_data():
    if not os.path.exists("shlassessments.csv"):
        st.warning("Warning: shlassessments.csv file not found. Using sample data instead.")
        return create_sample_data()

    df = pd.read_csv("shlassessments.csv", encoding='utf-8', encoding_errors='ignore')
    df.dropna(subset=['Name', 'URL'], inplace=True)
    df['combined_text'] = df.apply(lambda row: create_combined_text(row), axis=1)
    return df

def create_combined_text(row):
    fields = [
        str(row.get('Name', '')),
        str(row.get('Test type', '')),
        str(row.get('Description', '')),
        str(row.get('Keywords', '')),
        str(row.get('Skills measured', '')),
        str(row.get('Industries', ''))
    ]
    return ' '.join(fields).strip()

class VectorDB:
    def __init__(self):
        self.index = None
        self.df = None
        self.embeddings = None

    def create_index(self, df):
        self.df = df
        with st.spinner("Generating embeddings from Cohere..."):
            texts = df['combined_text'].tolist()
            response = co.embed(texts=texts, model="embed-english-v3.0", input_type="search_document")
            self.embeddings = np.array(response.embeddings).astype("float32")
            faiss.normalize_L2(self.embeddings)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

    def search(self, query, top_n=5):
        query_embedding = co.embed(texts=[query], model="embed-english-v3.0", input_type="search_query").embeddings
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_n)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                row = self.df.iloc[idx]
                explanation = self.generate_explanation(query, row)
                results.append({
                    "Assessment Name": row["Name"],
                    "URL": row["URL"],
                    "Duration": row.get("Duration", "Not specified"),
                    "Test Type": row.get("Test type", "Not specified"),
                    "Remote Support": row.get("Remote testing support", "Not specified"),
                    "Adaptive/IRT": row.get("Adaptive/IRT support", "Not specified"),
                    "Similarity Score": f"{scores[0][i]:.2f}",
                    "Skills Measured": row.get("Skills measured", "Not specified"),
                    "Explanation": explanation
                })
        return results

    def generate_explanation(self, query, assessment):
        assessment_name = assessment.get("Name", "Unnamed Assessment")
        skills = assessment.get("Skills measured", "")
        test_type = assessment.get("Test type", "")
        description = assessment.get("Description", "")

        prompt = f"""
        You are an AI assistant helping a hiring manager. Based on the job description below, explain why the assessment "{assessment_name}" is a suitable recommendation. Be brief and clear.
        Job Description:
        {query[:300]}...
        Additional Assessment Information:
        - Type: {test_type}
        - Skills measured: {skills}
        - Description: {description}
        Explain the reasoning:
        """

        try:
            with st.spinner("Generating explanation..."):
                response = co.generate(
                    model="command-xlarge",
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                return response.generations[0].text.strip()
        except Exception as e:
            return f"Error: {e}"

def create_sample_data():
    return pd.DataFrame({
        'Name': ['Numerical Reasoning Test', 'Verbal Reasoning Test'],
        'URL': ['https://example.com/test/1', 'https://example.com/test/2'],
        'Duration': ['30 minutes', '25 minutes'],
        'Test type': ['Cognitive', 'Cognitive'],
        'Remote testing support': ['Yes', 'No'],
        'Adaptive/IRT support': ['Yes', 'No'],
        'Skills measured': ['Mathematical ability', 'Critical thinking'],
        'Description': ['Tests numerical data analysis.', 'Assesses comprehension and reasoning.']
    })

def main():
    # Title section with styled container
    st.markdown('<div class="title-container"><h1>🧠 SHL Assessment Recommender</h1><p style="font-size: 1.2rem; margin-top: 10px;">Find the most suitable SHL assessments for your job roles with AI-powered semantic search</p></div>', unsafe_allow_html=True)
    
    df = load_data()
    vector_db = VectorDB()
    vector_db.create_index(df)
    search_interface(vector_db)

def search_interface(vector_db):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### Input Method")
        input_type = st.radio("", ["Text Query", "URL of Job Description"])
    
    with col2:
        user_query = ""
        if input_type == "Text Query":
            st.markdown("### Enter Job Description")
            user_query = st.text_area("", placeholder="Paste job description here...", height=200)
        else:
            st.markdown("### Enter Job Posting URL")
            input_url = st.text_input("", placeholder="https://example.com/job-posting")
            if input_url:
                extracted_text = extract_text_from_url(input_url)
                if extracted_text.startswith("Error:"):
                    st.error(extracted_text)
                else:
                    user_query = extracted_text
                    with st.expander("📄 View Extracted Job Description"):
                        st.write(extracted_text)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        top_n = st.slider("Number of recommendations:", 1, 10, 5)
    with col2:
        search_pressed = st.button("Find Matching Assessments", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    if search_pressed and user_query:
        with st.spinner("Finding the best assessment matches..."):
            results = vector_db.search(user_query, top_n=top_n)

        if results:
            st.markdown(f"<h3>📊 Found {len(results)} Matching Assessments</h3>", unsafe_allow_html=True)
            for i, res in enumerate(results, 1):
                st.markdown(f"""
                <div class="assessment-card">
                    <div class="assessment-header">
                        <h3>{i}. {res['Assessment Name']}</h3>
                    </div>
                    <div style="display: flex; gap: 30px;">
                        <div style="flex: 2;">
                            <p>🔗 <a href="{res['URL']}" target="_blank">Open Assessment</a></p>
                            <p>⏱️ <b>Duration:</b> {res['Duration']}</p>
                            <p>📊 <b>Test Type:</b> {res['Test Type']}</p>
                            <p>🧠 <b>Skills Measured:</b> {res['Skills Measured']}</p>
                            <div class="explanation-box">
                                <p><b>💡 Why this assessment matches your needs:</b></p>
                                <p>{res['Explanation']}</p>
                            </div>
                        </div>
                        <div style="flex: 1;">
                            <div class="assessment-score">
                                <p>Match Score: {res['Similarity Score']}</p>
                            </div>
                            <p><b>Remote Support:</b> {res['Remote Support']}</p>
                            <p><b>Adaptive/IRT Support:</b> {res['Adaptive/IRT']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matching assessments found.")
    elif search_pressed and not user_query:
        st.warning("Please enter a job description or URL first.")

def extract_text_from_url(url):
    try:
        with st.spinner("Extracting text from URL..."):
            article = Article(url)
            article.download()
            article.parse()
            return article.text
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    main()
