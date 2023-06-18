import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_files(directory, extension='.txt'):
    """Get a list of files with the specified extension in a directory."""
    file_list = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_list.append(file)
    return file_list

def read_file(file_path):
    """Read the contents of a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def vectorize_texts(texts):
    """Convert texts to TF-IDF vectors."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()
    return vectors

def calculate_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""
    similarity_score = cosine_similarity(vector_a.reshape(1, -1), vector_b.reshape(1, -1))[0][0]
    return similarity_score

def calculate_percentage(similarity_score):
    """Calculate plagiarism percentage based on similarity score."""
    percentage = round(similarity_score * 100, 2)
    return percentage

def check_plagiarism(files):
    """Check plagiarism between files in a directory."""
    results = set()
    texts = [read_file(file) for file in files]
    vectors = vectorize_texts(texts)
    
    for i in range(len(files)):
        file_a = files[i]
        text_a = texts[i]
        vector_a = vectors[i]
        
        for j in range(i+1, len(files)):
            file_b = files[j]
            text_b = texts[j]
            vector_b = vectors[j]
            
            sim_score = calculate_similarity(vector_a, vector_b)
            percentage = calculate_percentage(sim_score)
            sample_pair = sorted((file_a, file_b))
            result = (sample_pair[0], sample_pair[1], percentage)
            results.add(result)
    
    return results

directory_path = '.'  # Replace with the directory containing the text files
sample_files = get_files(directory_path)
plagiarism_results = check_plagiarism(sample_files)

for data in plagiarism_results:
    file_a, file_b, percentage = data
    print(f"{file_a} vs {file_b}: {percentage}% plagiarism")
