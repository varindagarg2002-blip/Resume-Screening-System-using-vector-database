import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_job_description(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

resume_folder = r"D:\Vector\resumes_pdf"
jd_file = r"D:\Vector\job_description.txt"

resume_texts = []
resume_names = []

for file in os.listdir(resume_folder):
    if file.lower().endswith(".pdf"):
        path = os.path.join(resume_folder, file)
        text = extract_text_from_pdf(path)
        if text.strip():
            resume_texts.append(text)
            resume_names.append(file)

model = SentenceTransformer("all-MiniLM-L6-v2")

resume_embeddings = model.encode(resume_texts)
jd_text = read_job_description(jd_file)
jd_embedding = model.encode([jd_text])

scores = cosine_similarity(resume_embeddings, jd_embedding)

print("\nResume Alignment with Job Description:\n")

for i, name in enumerate(resume_names):
    percentage = scores[i][0] * 100
    print(f"{name}  →  {percentage:.2f}%")
