import nltk
import logging
import sys
import re
import string
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# Setup (Phase 1, 2 & 3 simplified)
news_corpus = {
    101: "The tech giant recently unveiled its latest quantum computing processor, promising to solve complex cryptographic problems in mere seconds. Experts predict this could revolutionize data security globally.",
    102: "In a stunning upset, the underdog national team defeated the reigning world champions 3-1 in the final match of the tournament. The victory sparked widespread celebrations across the capital.",
    103: "Global markets experienced high volatility today as inflation rates hit a ten-year peak. Investors are closely monitoring the central bank's next move regarding interest rate hikes.",
    104: "A new study published in a leading medical journal highlights the significant health benefits of a Mediterranean diet, noting a substantial decrease in cardiovascular diseases among participants.",
    105: "Space exploration took a giant leap forward as the new autonomous rover successfully landed on the Martian surface. It has already begun transmitting high-resolution atmospheric data back to Earth."
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]

processed_corpus = {doc_id: preprocess_text(text) for doc_id, text in news_corpus.items()}

# PHASE 4 =====================================================================
# Vector Space Model (TF-IDF) and Cosine Similarity Ranking

# 1. Prepare data for Scikit-Learn
# TfidfVectorizer expects strings, so we join our perfectly cleaned tokens back into strings
doc_ids = list(processed_corpus.keys())
corpus_texts = [" ".join(tokens) for tokens in processed_corpus.values()]

# 2. Build the TF-IDF Matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

logging.info(f"TF-IDF Matrix successfully built. Shape: {tfidf_matrix.shape}")

# 3. Cosine Similarity Search Engine
def vector_space_search(query_string, top_k=3):
    """
    Processes a free-text query and ranks documents based on Cosine Similarity.
    """
    # Preprocess the query so it matches the index vocabulary exactly
    query_tokens = preprocess_text(query_string)
    if not query_tokens:
        return []
        
    query_processed_string = " ".join(query_tokens)
    
    # Vectorize the query using the fitted TF-IDF model
    query_vector = tfidf_vectorizer.transform([query_processed_string])
    
    # Calculate cosine similarity against all documents in the matrix
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort indices by highest similarity score
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    
    # Format and return the top_k results
    results = []
    for idx in ranked_indices:
        score = cosine_similarities[idx]
        if score > 0:  # Ignore documents with zero similarity
            results.append((doc_ids[idx], score))
            
    return results[:top_k]

# ==========================================
# PHASE 4 VERIFICATION TEST CODE BLOCK
# ==========================================
def verify_phase_4():
    print("\n--- Phase 4 Verification ---")
    
    print(f"Total vocabulary terms in TF-IDF Model: {len(tfidf_vectorizer.vocabulary_)}")
    
    # Test Query 1: Tech related
    q1 = "latest tech processor quantum"
    res1 = vector_space_search(q1)
    print(f"\nSearch Query: '{q1}'")
    for rank, (d_id, score) in enumerate(res1, 1):
        print(f" Rank {rank} | Doc ID: {d_id} | Score: {score:.4f}")
        
    # Test Query 2: Medical/Health
    q2 = "diet health decrease"
    res2 = vector_space_search(q2)
    print(f"\nSearch Query: '{q2}'")
    for rank, (d_id, score) in enumerate(res2, 1):
        print(f" Rank {rank} | Doc ID: {d_id} | Score: {score:.4f}")
        
    # Test Query 3: Something not in the dataset
    q3 = "aliens visiting from another galaxy"
    res3 = vector_space_search(q3)
    print(f"\nSearch Query: '{q3}'")
    print("Results:", res3)
    
    print("\n✅ Phase 4 completed successfully. Ready for Phase 5.")

if __name__ == "__main__":
    verify_phase_4()
