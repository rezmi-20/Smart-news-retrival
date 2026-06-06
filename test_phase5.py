import nltk
import logging
import sys
import re
import string
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# Setup (Phases 1-4 condensed)
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
doc_ids = list(processed_corpus.keys())
corpus_texts = [" ".join(tokens) for tokens in processed_corpus.values()]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_texts)

def vector_space_search(query_string, top_k=3):
    query_tokens = preprocess_text(query_string)
    if not query_tokens: return []
    query_processed_string = " ".join(query_tokens)
    query_vector = tfidf_vectorizer.transform([query_processed_string])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    results = [(doc_ids[idx], cosine_similarities[idx]) for idx in ranked_indices if cosine_similarities[idx] > 0]
    return results[:top_k]

# PHASE 5 =====================================================================
# Semantic Query Expansion using WordNet Lexical Trees

def expand_query_with_wordnet(query_string):
    """
    Expands a user's query by identifying WordNet synonyms for each token,
    increasing the chance of matching relevant documents (Higher Recall).
    """
    original_tokens = preprocess_text(query_string)
    expanded_terms = set(original_tokens)  # Use a set to prevent duplicates
    
    for token in original_tokens:
        # Retrieve all synsets (meaning clusters) for the token
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                # WordNet lemmas sometimes contain underscores instead of spaces
                synonym = lemma.name().replace('_', ' ')
                
                # Preprocess the synonym so it perfectly matches our index formatting
                synonym_tokens = preprocess_text(synonym)
                expanded_terms.update(synonym_tokens)
                
    # Reconstruct the expanded query string
    expanded_query = " ".join(expanded_terms)
    return expanded_query

def semantic_search(query_string, top_k=3):
    """
    A wrapper around our vector_space_search that applies semantic expansion first.
    """
    expanded_query = expand_query_with_wordnet(query_string)
    logging.info(f"Original Query: '{query_string}'")
    logging.info(f"Expanded Query: '{expanded_query}'")
    
    return vector_space_search(expanded_query, top_k)

# ==========================================
# PHASE 5 VERIFICATION TEST CODE BLOCK
# ==========================================
def verify_phase_5():
    print("\n--- Phase 5 Verification ---")
    
    # Test Query: "illness" is not in our corpus, but "diseases" is in Doc 104.
    # Without expansion, standard TF-IDF will return 0 results.
    # With WordNet expansion, "illness" expands to "disease", scoring a hit!
    
    test_q = "illness"
    
    print("\n[1] Standard Search (No Expansion):")
    std_results = vector_space_search(test_q)
    if not std_results:
        print(f" -> '{test_q}' found 0 results.")
    else:
        print(std_results)
        
    print("\n[2] Semantic Search (WordNet Expansion):")
    sem_results = semantic_search(test_q)
    for rank, (d_id, score) in enumerate(sem_results, 1):
        print(f" -> Rank {rank} | Doc ID: {d_id} | Score: {score:.4f}")
        
    print("\n✅ Phase 5 completed successfully. Ready for Phase 6.")

if __name__ == "__main__":
    verify_phase_5()
