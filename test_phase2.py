import nltk
import logging
import sys
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# PHASE 1 =====================================================================
news_corpus = {
    101: "The tech giant recently unveiled its latest quantum computing processor, promising to solve complex cryptographic problems in mere seconds. Experts predict this could revolutionize data security globally.",
    102: "In a stunning upset, the underdog national team defeated the reigning world champions 3-1 in the final match of the tournament. The victory sparked widespread celebrations across the capital.",
    103: "Global markets experienced high volatility today as inflation rates hit a ten-year peak. Investors are closely monitoring the central bank's next move regarding interest rate hikes.",
    104: "A new study published in a leading medical journal highlights the significant health benefits of a Mediterranean diet, noting a substantial decrease in cardiovascular diseases among participants.",
    105: "Space exploration took a giant leap forward as the new autonomous rover successfully landed on the Martian surface. It has already begun transmitting high-resolution atmospheric data back to Earth."
}

# PHASE 2 =====================================================================
# Text Cleaning, Pipeline Tokenization, and Normalization Module

# Initialize the lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans, tokenizes, and normalizes the input text.
    - Lowercase
    - Remove punctuation and numbers
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    # Replace anything that isn't a letter with a space
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords and 5. Lemmatize
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    
    return clean_tokens

# Apply the preprocessing pipeline to the entire corpus
processed_corpus = {}
for doc_id, text in news_corpus.items():
    processed_corpus[doc_id] = preprocess_text(text)
    logging.info(f"Processed Document {doc_id}. Token count: {len(processed_corpus[doc_id])}")

# ==========================================
# PHASE 2 VERIFICATION TEST CODE BLOCK
# ==========================================
def verify_phase_2():
    print("\n--- Phase 2 Verification ---")
    
    if not isinstance(processed_corpus, dict):
        print("[FAILED] processed_corpus is not a dictionary.")
        return
        
    print(f"[SUCCESS] Processed all {len(processed_corpus)} documents.")
    
    first_doc_id = list(processed_corpus.keys())[0]
    print("\n--- Sample Document Processing ---")
    print(f"Original (Doc {first_doc_id}): {news_corpus[first_doc_id]}")
    print(f"Tokens (Doc {first_doc_id}): {processed_corpus[first_doc_id]}")
    
    print("\nPhase 2 completed successfully. Ready for Phase 3.")

verify_phase_2()
