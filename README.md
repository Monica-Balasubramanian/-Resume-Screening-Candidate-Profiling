Resume-Screening-Candidate-Profiling



import streamlit as st
import pandas as pd
import fitz 
import docx
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

skill_db = ["Python", "Machine Learning", "Data Science", "SQL", "TensorFlow", "JavaScript", "Java", "Docker", "AWS", "Deep Learning"]

def set_background():
    """Set a background image for the app."""
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("https://thumbs.dreamstime.com/z/hr-candidate-selection-staffing-talent-background-choice-business-career-hr-candidate-selection-staffing-talent-background-choice-148322873.jpg");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF files using PyMuPDF."""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(uploaded_file):
    """Extract text from DOCX files using python-docx."""
    try:
        doc = docx.Document(uploaded_file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def clean_text(text):
    """Preprocess text: remove punctuation, lowercase, and stopwords."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

def extract_skills(text):
    """Extract skills by keyword matching."""
    skills = [skill for skill in skill_db if skill.lower() in text.lower()]
    return list(set(skills))

def match_resumes_to_job(resume_texts, job_desc):
    """Compute similarity scores using TF-IDF & Cosine Similarity."""
    vectorizer = TfidfVectorizer()
    corpus = resume_texts + [job_desc]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])
    return similarity_scores.flatten()

def main():
    set_background() 

    with st.sidebar:
        st.header("📌 Job Description & Resumes")
        job_desc = st.text_area("📝 Enter Job Description:")
        uploaded_files = st.file_uploader("📂 Upload Resumes (PDF/DOCX)", accept_multiple_files=True, type=["pdf", "docx"])

    st.title("📄 Resume Screening & Candidate Profiling")
    
    col1, col2 = st.columns([1, 2])

    with col2:  
        if st.button("🚀 Process & Rank Resumes"):
            if job_desc and uploaded_files:
                resume_texts = []
                resume_data = []
                
                for file in uploaded_files:
                    file_extension = file.name.split(".")[-1]
                    text = ""

                    if file_extension == "pdf":
                        text = extract_text_from_pdf(file)
                    elif file_extension == "docx":
                        text = extract_text_from_docx(file)

                    if text:
                        cleaned_text = clean_text(text)
                        skills = extract_skills(cleaned_text)
                        resume_texts.append(cleaned_text)
                        resume_data.append({"Name": file.name, "Skills": skills})
                
                scores = match_resumes_to_job(resume_texts, job_desc)

                for i, score in enumerate(scores):
                    resume_data[i]["Match Score"] = round(score * 100, 2)

                df = pd.DataFrame(resume_data).sort_values(by="Match Score", ascending=False)
                
                st.subheader("🏆 Ranked Candidates")
                st.dataframe(df)
            else:
                st.warning("⚠️ Please enter a job description and upload resumes.")

if __name__ == "__main__":
    main()
