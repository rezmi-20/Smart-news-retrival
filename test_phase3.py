import nltk
import logging
import sys
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# PHASE 1 & 2 Setup ==========================================================
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
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return clean_tokens

processed_corpus = {doc_id: preprocess_text(text) for doc_id, text in news_corpus.items()}

# PHASE 3 =====================================================================
# Custom Inverted Index Construction and Boolean Query Engine

# 1. Build the Inverted Index
inverted_index = defaultdict(set)

for doc_id, tokens in processed_corpus.items():
    for token in set(tokens): # Use set to avoid adding same doc_id multiple times unnecessarily
        inverted_index[token].add(doc_id)

logging.info(f"Built Inverted Index with {len(inverted_index)} unique terms.")

# 2. Boolean Query Engine
def boolean_search(query_string):
    """
    Evaluates a left-to-right boolean query.
    Supported operators: AND, OR, NOT.
    """
    tokens = query_string.split()
    if not tokens:
        return set()
    
    all_docs = set(processed_corpus.keys())
    
    def fetch_term_docs(term):
        # Preprocess the term exactly how the corpus was processed
        pt = preprocess_text(term)
        if not pt: 
            return set()
        # Return a copy of the set from the inverted index
        return set(inverted_index.get(pt[0], set()))

    result = None
    current_operator = None
    invert_next = False
    
    for token in tokens:
        upper_tok = token.upper()
        if upper_tok == 'AND':
            current_operator = 'AND'
        elif upper_tok == 'OR':
            current_operator = 'OR'
        elif upper_tok == 'NOT':
            invert_next = True
        else:
            # It's a standard search term
            term_docs = fetch_term_docs(token)
            
            # Apply NOT if flagged
            if invert_next:
                term_docs = all_docs - term_docs
                invert_next = False
                
            # Combine with result so far
            if result is None:
                result = term_docs
            else:
                if current_operator == 'AND':
                    result = result.intersection(term_docs)
                elif current_operator == 'OR':
                    result = result.union(term_docs)
                current_operator = None
                
    return result if result is not None else set()

# ==========================================
# PHASE 3 VERIFICATION TEST CODE BLOCK
# ==========================================
def verify_phase_3():
    print("\n--- Phase 3 Verification ---")
    
    print(f"[SUCCESS] Inverted Index Size: {len(inverted_index)} distinct terms.")
    
    # Test 1: Simple term
    q1 = "quantum"
    res1 = boolean_search(q1)
    print(f"Query: '{q1}' | Result: {res1}")
    assert res1 == {101}, "Failed Test 1"
    
    # Test 2: AND operator
    q2 = "global AND market"
    res2 = boolean_search(q2)
    print(f"Query: '{q2}' | Result: {res2}")
    assert res2 == {103}, "Failed Test 2"
    
    # Test 3: OR operator
    q3 = "martian OR medical"
    res3 = boolean_search(q3)
    print(f"Query: '{q3}' | Result: {res3}")
    assert res3 == {104, 105}, "Failed Test 3"
    
    # Test 4: NOT operator
    q4 = "NOT quantum"
    res4 = boolean_search(q4)
    print(f"Query: '{q4}' | Result: {res4}")
    assert res4 == {102, 103, 104, 105}, "Failed Test 4"

    # Test 5: Complex query
    q5 = "market AND NOT medical"
    res5 = boolean_search(q5)
    print(f"Query: '{q5}' | Result: {res5}")
    assert res5 == {103}, "Failed Test 5"
    
    print("\n✅ Phase 3 completed successfully. Ready for Phase 4.")

if __name__ == "__main__":
    verify_phase_3()
