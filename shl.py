import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from newspaper import Article
import nltk
from nltk.corpus import stopwords
import faiss
from sentence_transformers import SentenceTransformer
import cohere
import requests
import zipfile
# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Enhanced SHL Assessment Recommender",
    page_icon="🔍",
    layout="wide"
)
def download_model():
    if not os.path.exists("multi-qa-MiniLM-L6-cos-v1"):
        url = "https://drive.google.com/uc?id=1ZPXTAZCy4fmGgeCqdG5AUKSqo4TUFMpy&export=download"  # Direct download link
        try:
            # Download the file
            response = requests.get(url, timeout=120)
            if response.status_code != 200:
                raise Exception(f"Failed to download file: HTTP {response.status_code}")
            
            # Save the file
            with open("model.zip", "wb") as f:
                f.write(response.content)
            
            # Verify the downloaded file
            with zipfile.ZipFile("model.zip", "r") as zip_ref:
                zip_ref.extractall(".")
            
            os.remove("model.zip")  # Clean up

        except zipfile.BadZipFile:
            print("Error: The downloaded file is not a valid zip file.")
            if os.path.exists("model.zip"):
                os.remove("model.zip")  # Delete corrupted file
        except Exception as e:
            print(f"An error occurred: {e}")

download_model()
# Set up Cohere API key directly in the code
COHERE_API_KEY = "j7uTsOOCsQS99XqLFRUFWHWzWQzADufa6AuHWxXU"  # Replace with your actual API key

@st.cache_resource
def load_model():
    """Load the sentence transformer model for embeddings"""
    download_model()
    return SentenceTransformer("./multi-qa-MiniLM-L6-cos-v1")

@st.cache_data
def load_data():
    """Load assessment data from CSV or use sample data"""
    if not os.path.exists("shlassessments.csv"):
        st.warning("Warning: shlassessments.csv file not found. Using sample data instead.")
        return create_sample_data()

    df = pd.read_csv("shlassessments.csv", encoding='utf-8', encoding_errors='ignore')
    df.dropna(subset=['Name', 'URL'], inplace=True)
    df['combined_text'] = df.apply(lambda row: create_combined_text(row), axis=1)
    return df

def create_combined_text(row):
    """Create a combined text description from all relevant fields for better semantic matching"""
    fields = []
    fields.append(str(row['Name']) if 'Name' in row else '')
    fields.append(str(row['Test type']) if 'Test type' in row else '')

    for field in ['Description', 'Keywords', 'Skills measured', 'Industries']:
        if field in row and pd.notna(row[field]):
            fields.append(str(row[field]))

    return ' '.join(fields).strip()

class VectorDB:
    def __init__(self, model):
        self.model = model
        self.index = None
        self.df = None
        self.embeddings = None

    def create_index(self, df):
        """Create FAISS index from dataframe"""
        self.df = df
        with st.spinner("Creating embeddings for assessments. This may take a moment..."):
            self.embeddings = self.model.encode(df['combined_text'].tolist())
            faiss.normalize_L2(self.embeddings)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

    def search(self, query, top_n=5):
        """Search the index for most similar assessments"""
        query_embedding = self.model.encode([query])
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
        """Generate reasoning for why this assessment matches the job description using Cohere API"""
        co = cohere.Client(COHERE_API_KEY)
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
                    model="command-xlarge",  # Correct model for `generate` API
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7
                )
                explanation = response.generations[0].text.strip()
                return explanation
        except Exception as e:
            return f"Error: {str(e)}"

def create_sample_data():
    """Create sample data if shlassessments.csv is not available"""
    sample_data = {
        'Name': ['Numerical Reasoning Test', 'Verbal Reasoning Test'],
        'URL': ['https://example.com/test/1', 'https://example.com/test/2'],
        'Duration': ['30 minutes', '25 minutes'],
        'Test type': ['Cognitive', 'Cognitive'],
        'Remote testing support': ['Yes', 'No'],
        'Adaptive/IRT support': ['Yes', 'No'],
        'Skills measured': ['Mathematical ability', 'Critical thinking'],
        'Description': ['Tests numerical data analysis.', 'Assesses comprehension and reasoning.']
    }
    return pd.DataFrame(sample_data)

def main():
    st.title("🔍 Enhanced SHL Assessment Recommender")
    st.write("Find the most suitable SHL assessments based on job descriptions with semantic search.")
    model = load_model()
    df = load_data()
    vector_db = VectorDB(model)
    vector_db.create_index(df)
    search_interface(vector_db)

def search_interface(vector_db):
    input_type = st.radio("Choose Input Type:", ["Text Query", "URL of Job Description"])
    user_query = ""

    if input_type == "Text Query":
        user_query = st.text_area("Enter job description or query:", height=200)
    else:
        input_url = st.text_input("Enter job posting URL:")
        if input_url:
            extracted_text = extract_text_from_url(input_url)
            if extracted_text.startswith("Error:"):
                st.error(extracted_text)
            else:
                user_query = extracted_text
                with st.expander("Extracted Job Description"):
                    st.write(extracted_text)

    top_n = st.slider("Number of recommendations:", 1, 10, 5)
    search_pressed = st.button("Find Matching Assessments")

    if search_pressed and user_query:
        with st.spinner("Finding the best assessment matches..."):
            results = vector_db.search(user_query, top_n=top_n)

        if results:
            st.success(f"Found {len(results)} matching assessments!")
            for i, res in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([2, 1])  # Create two-column layout
                    with col1:
                        st.subheader(f"{i}. {res['Assessment Name']}")
                        st.write(f"🔗 [Open Assessment]({res['URL']})")
                        st.write(f"⏱️ Duration: {res['Duration']}")
                        st.write(f"📊 Type: {res['Test Type']}")
                        st.write(f"🧠 Skills Measured: {res['Skills Measured']}")
                        st.info(f"💡 **Why this assessment matches your needs**: {res['Explanation']}")
                    with col2:
                        st.write(f"**Match Score**: {res['Similarity Score']}")
                        st.write(f"**Remote Support**: {res['Remote Support']}")
                        st.write(f"**Adaptive/IRT Support**: {res['Adaptive/IRT']}")
        else:
            st.warning("No matching assessments found.")
    elif search_pressed and not user_query:
        st.warning("Please enter a job description or URL first.")

def extract_text_from_url(url):
    """Extract and process text content from a URL"""
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
