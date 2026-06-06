import streamlit as st
import re
import numpy as np
import requests
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. UI CONFIG & PREMIUM CSS ---
st.set_page_config(page_title="the news dispatch.", page_icon="📰", layout="wide")

# Injecting Premium Typography and Layout CSS into Streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,700&family=Inter:wght@400;500;600&display=swap');

/* Hide Streamlit Chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Background & Fonts */
.stApp {
    background-color: #f8f6f0;
    font-family: 'Inter', sans-serif;
    color: #1a1a1a;
}

/* Custom Header Logo */
.news-logo {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 30px;
    margin-top: 20px;
    color: #1a1a1a;
}

/* Beautiful Cards */
.editorial-card {
    background: transparent;
    border-radius: 8px;
    transition: transform 0.2s ease, opacity 0.2s ease;
    margin-bottom: 25px;
}
.editorial-card:hover {
    transform: translateY(-5px);
    opacity: 0.9;
}
.editorial-img {
    width: 100%;
    border-radius: 8px;
    object-fit: cover;
    aspect-ratio: 16/10;
    margin-bottom: 10px;
}
.editorial-cat {
    color: #c0392b;
    text-transform: uppercase;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
}
.editorial-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    margin: 5px 0 0 0;
    line-height: 1.2;
    font-weight: 700;
}
.editorial-title a {
    color: #1a1a1a;
    text-decoration: none;
}
.editorial-score {
    font-size: 0.8rem;
    color: #666;
    font-style: italic;
    margin-top: 5px;
}
</style>
<div class="news-logo">the news<br>dispatch.</div>
""", unsafe_allow_html=True)

# --- 1. SETUP & LIVE DATA INGESTION ---
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()

# Pre-defined high-quality Unsplash images for instant loading (fixes efficiency issue)
CATEGORY_IMAGES = {
    'WORLD': 'https://images.unsplash.com/photo-1521295121783-8a321d551ad2?auto=format&fit=crop&q=80&w=800',
    'NATION': 'https://images.unsplash.com/photo-1461896836934-ffe607ba8211?auto=format&fit=crop&q=80&w=800',
    'BUSINESS': 'https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&q=80&w=800',
    'TECHNOLOGY': 'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=800',
    'LATEST': 'https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=800'
}

@st.cache_data(ttl=3600)
def fetch_live_news():
    urls = [
        'https://news.google.com/rss',
        'https://news.google.com/rss/headlines/section/topic/WORLD',
        'https://news.google.com/rss/headlines/section/topic/NATION',
        'https://news.google.com/rss/headlines/section/topic/BUSINESS',
        'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY',
    ]
    corpus = {}
    doc_id = 1
    for url in urls:
        try:
            res = requests.get(url, timeout=10)
            root = ET.fromstring(res.content)
            cat = url.split('/')[-1] if 'topic' in url else 'LATEST'
            for item in root.findall('.//item')[:30]:
                title = item.find('title').text
                link = item.find('link').text
                image = CATEGORY_IMAGES.get(cat, CATEGORY_IMAGES['LATEST'])
                corpus[doc_id] = {'title': title, 'link': link, 'category': cat, 'image': image}
                doc_id += 1
        except Exception:
            pass
    return corpus

with st.spinner("Fetching live dispatch..."):
    news_corpus = fetch_live_news()

if not news_corpus:
    st.error("Failed to fetch live news. Please check your internet connection.")
    st.stop()

# --- 2. PIPELINE TEXT PROCESSING ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = {'vs', 'us', 'usa', 'uk'}
stop_words.update(custom_stopwords)

@st.cache_data
def build_index(corpus):
    processed = {}
    for doc_id, article in corpus.items():
        text = article['title'].lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        clean = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
        processed[doc_id] = clean
        
    doc_ids = list(processed.keys())
    corpus_texts = [" ".join(tokens) for tokens in processed.values()]
    
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus_texts)
    return doc_ids, vectorizer, matrix

doc_ids, tfidf_vectorizer, tfidf_matrix = build_index(news_corpus)

def preprocess_query(query_string):
    text = query_string.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]

# --- 3. VECTOR SPACE MODEL (TF-IDF) ---
def vector_space_search(query_string, top_k=12):
    query_tokens = preprocess_query(query_string)
    if not query_tokens: return []
    query_processed_string = " ".join(query_tokens)
    query_vector = tfidf_vectorizer.transform([query_processed_string])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = np.argsort(cosine_similarities)[::-1]
    results = [(doc_ids[idx], cosine_similarities[idx]) for idx in ranked_indices if cosine_similarities[idx] > 0]
    return results[:top_k]

# --- 4. SEMANTIC EXPANSION ---
def expand_query_with_wordnet(query_string):
    original_tokens = preprocess_query(query_string)
    expanded_terms = set(original_tokens)
    for token in original_tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                expanded_terms.update(preprocess_query(synonym))
    return " ".join(expanded_terms)

def semantic_search(query_string, top_k=12):
    expanded_query = expand_query_with_wordnet(query_string)
    return vector_space_search(expanded_query, top_k), expanded_query

# --- 5. STREAMLIT FRONTEND UI ---
st.markdown(f"<div style='text-align: center; color: #666; margin-bottom: 20px;'>Indexing {len(news_corpus)} articles</div>", unsafe_allow_html=True)

# Using st.form so hitting "Enter" works!
with st.form(key='search_form'):
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Search news...", placeholder="e.g., technology, economy, world", label_visibility="collapsed")
    with col2:
        submit_button = st.form_submit_button(label='Search Dispatch')
    
    use_semantic = st.checkbox("Enable Semantic WordNet Expansion")

if submit_button:
    if search_query:
        if use_semantic:
            results, expanded_q = semantic_search(search_query)
            st.info(f"**Expanded Semantic Engine used:** `{expanded_q}`")
        else:
            results = vector_space_search(search_query)
            
        if results:
            st.success(f"Found {len(results)} relevant documents.")
            
            # Create a 3-column grid for the premium masonry feel
            cols = st.columns(3)
            for idx, (d_id, score) in enumerate(results):
                article = news_corpus[d_id]
                col = cols[idx % 3]
                with col:
                    st.markdown(f"""
                    <div class="editorial-card">
                        <img src="{article['image']}" class="editorial-img">
                        <div class="editorial-cat">{article['category']}</div>
                        <h3 class="editorial-title"><a href="{article['link']}" target="_blank">{article['title']}</a></h3>
                        <div class="editorial-score">Relevance: {score:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No relevant documents found. Try enabling Semantic Expansion!")
    else:
        st.error("Please enter a search query.")
