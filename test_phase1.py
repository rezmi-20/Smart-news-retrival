import nltk
import logging
import sys

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)

# 2. Download required NLTK datasets
def download_nltk_data():
    nltk_packages = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']
    for pkg in nltk_packages:
        try:
            nltk.download(pkg, quiet=True)
            logging.info(f"Successfully downloaded/verified NLTK package: '{pkg}'")
        except Exception as e:
            logging.error(f"Failed to download NLTK package '{pkg}': {e}")

download_nltk_data()

# 3. Define the mock news corpus
news_corpus = {
    101: "The tech giant recently unveiled its latest quantum computing processor, promising to solve complex cryptographic problems in mere seconds. Experts predict this could revolutionize data security globally.",
    102: "In a stunning upset, the underdog national team defeated the reigning world champions 3-1 in the final match of the tournament. The victory sparked widespread celebrations across the capital.",
    103: "Global markets experienced high volatility today as inflation rates hit a ten-year peak. Investors are closely monitoring the central bank's next move regarding interest rate hikes.",
    104: "A new study published in a leading medical journal highlights the significant health benefits of a Mediterranean diet, noting a substantial decrease in cardiovascular diseases among participants.",
    105: "Space exploration took a giant leap forward as the new autonomous rover successfully landed on the Martian surface. It has already begun transmitting high-resolution atmospheric data back to Earth."
}

logging.info(f"Successfully constructed news_corpus with {len(news_corpus)} articles.")

# ==========================================
# PHASE 1 VERIFICATION TEST CODE BLOCK
# ==========================================
def verify_phase_1():
    print("--- Phase 1 Verification ---")
    
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        sw = stopwords.words('english')
        print(f"[SUCCESS] Loaded {len(sw)} English stopwords from NLTK.")
        
        sample_tokens = word_tokenize("Information Retrieval is fascinating!")
        print(f"[SUCCESS] Tokenization test passed: {sample_tokens}")
        
    except LookupError as e:
        print(f"[FAILED] NLTK data missing: {e}")
        return

    if not isinstance(news_corpus, dict):
        print("[FAILED] news_corpus is not a dictionary.")
        return
        
    if len(news_corpus) < 5:
        print(f"[FAILED] news_corpus has insufficient documents: {len(news_corpus)} found.")
        return
        
    print(f"\n[SUCCESS] Loaded {len(news_corpus)} distinct articles in the corpus.")
    print("\n--- Sample Document Preview ---")
    
    first_doc_id = list(news_corpus.keys())[0]
    print(f"Doc ID: {first_doc_id}")
    print(f"Content: {news_corpus[first_doc_id]}")
    
    print("\n✅ Phase 1 completed successfully.")

verify_phase_1()
