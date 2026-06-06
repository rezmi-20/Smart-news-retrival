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

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The News Dispatch",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS INJECTION
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Root palette ── */
:root {
  --cream: #f5f0e8;
  --ink:   #111111;
  --red:   #c0392b;
  --mid:   #888888;
  --card-bg: #ffffff;
  --border: #e2ddd5;
}

/* ── Page background ── */
.stApp { background: var(--cream) !important; }

/* ── Masthead ── */
.masthead {
  border-bottom: 3px double var(--ink);
  text-align: center;
  padding: 32px 40px 16px;
  margin-bottom: 0;
}
.masthead-date {
  font-family: 'Inter', sans-serif;
  font-size: 0.78rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--mid);
  margin-bottom: 8px;
}
.masthead-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.6rem, 6vw, 4.8rem);
  font-weight: 700;
  color: var(--ink);
  line-height: 1;
  letter-spacing: -2px;
  margin: 0;
}
.masthead-tagline {
  font-family: 'Inter', sans-serif;
  font-size: 0.8rem;
  color: var(--mid);
  margin-top: 8px;
  letter-spacing: 1px;
}

/* ── Thin rule below masthead ── */
.rule-thin {
  border: none;
  border-top: 1px solid var(--ink);
  margin: 6px 40px;
}

/* ── Search bar ── */
.search-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 40px;
  border-bottom: 1px solid var(--border);
  background: var(--cream);
}

/* ── Category pills ── */
.category-bar {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 0 40px 16px;
  border-bottom: 2px solid var(--ink);
}
.cat-pill {
  font-family: 'Inter', sans-serif;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 4px 14px;
  border-radius: 20px;
  border: 1px solid var(--ink);
  cursor: pointer;
  background: transparent;
  color: var(--ink);
  transition: background 0.2s, color 0.2s;
}
.cat-pill:hover { background: var(--ink); color: var(--cream); }
.cat-pill.active { background: var(--ink); color: var(--cream); }

/* ── Grid ── */
.news-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2px;
  padding: 2px 40px 40px;
  margin-top: 18px;
}
.news-grid.has-hero {
  grid-template-columns: repeat(3, 1fr);
}

/* ── Cards ── */
.ncard {
  background: var(--card-bg);
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: box-shadow 0.25s ease, transform 0.25s ease;
  border: 1px solid var(--border);
}
.ncard:hover {
  box-shadow: 0 8px 32px rgba(0,0,0,0.13);
  transform: translateY(-4px);
  z-index: 10;
}
.ncard.hero {
  grid-column: span 2;
}
.ncard img {
  width: 100%;
  aspect-ratio: 16/9;
  object-fit: cover;
  display: block;
}
.ncard.hero img { aspect-ratio: 21/9; }
.ncard-body {
  padding: 16px 18px 18px;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ncard-cat {
  font-family: 'Inter', sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--red);
}
.ncard-title {
  font-family: 'Playfair Display', serif;
  font-weight: 700;
  font-size: 1.05rem;
  line-height: 1.3;
  color: var(--ink);
  margin: 0;
}
.ncard.hero .ncard-title { font-size: 1.6rem; }
.ncard-link {
  display: block;
  margin-top: auto;
  padding-top: 10px;
  font-family: 'Inter', sans-serif;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--red);
  text-decoration: none;
  letter-spacing: 0.5px;
}
.ncard-score {
  font-family: 'Inter', sans-serif;
  font-size: 0.7rem;
  color: var(--mid);
  font-style: italic;
}

/* ── Streamlit input override ── */
.stTextInput > div > div > input {
  background: #fff !important;
  border: 1.5px solid var(--ink) !important;
  border-radius: 2px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.95rem !important;
  color: var(--ink) !important;
  padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--red) !important;
  box-shadow: none !important;
}
.stButton > button {
  background: var(--ink) !important;
  color: var(--cream) !important;
  border: none !important;
  border-radius: 2px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
  letter-spacing: 1px !important;
  padding: 10px 22px !important;
  transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--red) !important; }

/* ── Status messages ── */
.status-bar {
  padding: 10px 40px;
  font-family: 'Inter', sans-serif;
  font-size: 0.8rem;
  color: var(--mid);
  border-bottom: 1px solid var(--border);
}
.no-results {
  text-align: center;
  padding: 60px 40px;
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem;
  font-style: italic;
  color: var(--mid);
}
.expanded-pill {
  background: #fff3f3;
  border-left: 3px solid var(--red);
  padding: 8px 16px;
  margin: 0 40px 12px;
  font-family: 'Inter', sans-serif;
  font-size: 0.8rem;
  color: var(--mid);
}

/* ── Loader ── */
.stSpinner > div { border-top-color: var(--red) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  NLTK SETUP
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        nltk.download(pkg, quiet=True)

download_nltk()

# ─────────────────────────────────────────────
#  CATEGORY IMAGE POOL  (curated, varied)
# ─────────────────────────────────────────────
import random
CAT_IMGS = {
    'WORLD': [
        'https://images.unsplash.com/photo-1491336477066-31156b5e4f35?w=800&q=80',
        'https://images.unsplash.com/photo-1522199710521-72d69614c702?w=800&q=80',
        'https://images.unsplash.com/photo-1521295121783-8a321d551ad2?w=800&q=80',
    ],
    'NATION': [
        'https://images.unsplash.com/photo-1529107386315-e1a2ed48a620?w=800&q=80',
        'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80',
    ],
    'BUSINESS': [
        'https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?w=800&q=80',
        'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=800&q=80',
        'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80',
    ],
    'TECHNOLOGY': [
        'https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80',
        'https://images.unsplash.com/photo-1531297484001-80022131f5a1?w=800&q=80',
        'https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=800&q=80',
    ],
    'LATEST': [
        'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80',
        'https://images.unsplash.com/photo-1585829365295-ab7cd400c167?w=800&q=80',
    ],
}
def get_image(cat):
    pool = CAT_IMGS.get(cat, CAT_IMGS['LATEST'])
    return random.choice(pool)

# ─────────────────────────────────────────────
#  LIVE NEWS FETCHER
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_news():
    feeds = [
        ('LATEST',     'https://news.google.com/rss'),
        ('WORLD',      'https://news.google.com/rss/headlines/section/topic/WORLD'),
        ('NATION',     'https://news.google.com/rss/headlines/section/topic/NATION'),
        ('BUSINESS',   'https://news.google.com/rss/headlines/section/topic/BUSINESS'),
        ('TECHNOLOGY', 'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY'),
        ('SPORTS',     'https://news.google.com/rss/headlines/section/topic/SPORTS'),
        ('HEALTH',     'https://news.google.com/rss/headlines/section/topic/HEALTH'),
    ]
    corpus = {}
    doc_id = 1
    for cat, url in feeds:
        try:
            res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(res.content)
            for item in root.findall('.//item')[:25]:
                title_el = item.find('title')
                link_el  = item.find('link')
                if title_el is None or link_el is None:
                    continue
                raw_title = title_el.text or ''
                # Strip " - Source Name" suffix from Google News titles
                title = re.sub(r'\s*-\s*[^-]+$', '', raw_title).strip()
                corpus[doc_id] = {
                    'title':    title,
                    'raw':      raw_title,
                    'link':     link_el.text,
                    'category': cat,
                    'image':    get_image(cat),
                }
                doc_id += 1
        except Exception:
            pass
    return corpus

# ─────────────────────────────────────────────
#  NLP  &  INDEX
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
_stop = set(stopwords.words('english')) | {'vs', 'say', 'said', 'says', 'uk', 'mr', 'new'}

def _clean(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in _stop and len(w) > 2]

@st.cache_data(show_spinner=False)
def build_index(corpus_keys, corpus_vals):
    """
    Build TF-IDF index.
    IMPORTANT FIX: document text = title tokens + category keyword repeated
    so category-level queries like 'technology' always match.
    """
    doc_ids, texts = [], []
    for did, article in zip(corpus_keys, corpus_vals):
        tokens = _clean(article['title'])
        # Append category as extra searchable tokens (fixes the 'technology' bug)
        tokens += _clean(article['category'])
        doc_ids.append(did)
        texts.append(" ".join(tokens))

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    return doc_ids, vec, mat

def preprocess_query(q: str) -> str:
    return " ".join(_clean(q))

# ─────────────────────────────────────────────
#  SEARCH FUNCTIONS
# ─────────────────────────────────────────────
def tfidf_search(query: str, corpus, doc_ids, vectorizer, matrix, top_k=12):
    pq = preprocess_query(query)
    if not pq:
        return []
    qvec = vectorizer.transform([pq])
    sims = cosine_similarity(qvec, matrix).flatten()
    ranked = np.argsort(sims)[::-1]
    return [(doc_ids[i], float(sims[i])) for i in ranked if sims[i] > 0.001][:top_k]

def semantic_expand(query: str) -> str:
    tokens = _clean(query)
    expanded = set(tokens)
    for tok in tokens:
        for syn in wordnet.synsets(tok):
            # Only use first 2 synsets to avoid explosion
            for lemma in list(syn.lemmas())[:3]:
                expanded.update(_clean(lemma.name().replace('_', ' ')))
    return " ".join(expanded)

# ─────────────────────────────────────────────
#  CARD HTML
# ─────────────────────────────────────────────
def card_html(article, score=None, hero=False):
    hero_cls = 'hero' if hero else ''
    score_html = f'<div class="ncard-score">Relevance: {score:.3f}</div>' if score else ''
    # Clean up title for display (strip trailing source " - XYZ")
    display_title = article['title']
    return f"""
<div class="ncard {hero_cls}">
  <img src="{article['image']}" alt="news" loading="lazy"
       onerror="this.src='https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80'">
  <div class="ncard-body">
    <span class="ncard-cat">{article['category']}</span>
    <h3 class="ncard-title">{display_title}</h3>
    {score_html}
    <a class="ncard-link" href="{article['link']}" target="_blank">Read full story →</a>
  </div>
</div>"""

# ─────────────────────────────────────────────
#  FETCH & INDEX  (with loading state)
# ─────────────────────────────────────────────
with st.spinner("Fetching today's dispatch..."):
    news_corpus = fetch_live_news()

if not news_corpus:
    st.error("Could not fetch live news. Check your internet connection.")
    st.stop()

keys  = list(news_corpus.keys())
vals  = list(news_corpus.values())
doc_ids, tfidf_vec, tfidf_mat = build_index(tuple(keys), tuple(vals))

# ─────────────────────────────────────────────
#  MASTHEAD
# ─────────────────────────────────────────────
from datetime import datetime
today = datetime.now().strftime("%A, %B %d, %Y")

st.markdown(f"""
<div class="masthead">
  <div class="masthead-date">{today}</div>
  <h1 class="masthead-title">The News Dispatch</h1>
  <div class="masthead-tagline">Live · Intelligent · Searchable</div>
</div>
<hr class="rule-thin">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SEARCH BAR
# ─────────────────────────────────────────────
with st.form(key='search_form', clear_on_submit=False):
    c1, c2, c3 = st.columns([5, 1, 1.2])
    with c1:
        query = st.text_input("query", placeholder="Search today's news — e.g.  iran, technology, economy ...",
                              label_visibility="collapsed")
    with c2:
        submitted = st.form_submit_button("Search")
    with c3:
        use_semantic = st.checkbox("Semantic Expansion", value=False)

# ─────────────────────────────────────────────
#  CATEGORY PILLS  (visual only)
# ─────────────────────────────────────────────
cats = ['ALL', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'SPORTS', 'HEALTH']
pills = "".join(f'<span class="cat-pill{"  active" if c=="ALL" else ""}">{c}</span>' for c in cats)
st.markdown(f'<div class="category-bar">{pills}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  RESULTS LOGIC
# ─────────────────────────────────────────────
if submitted and query.strip():
    expanded_q = None
    if use_semantic:
        expanded_q = semantic_expand(query)
        search_q   = expanded_q
    else:
        search_q = query

    results = tfidf_search(search_q, news_corpus, doc_ids, tfidf_vec, tfidf_mat)

    if expanded_q:
        st.markdown(f'<div class="expanded-pill">Semantic expansion: {expanded_q}</div>',
                    unsafe_allow_html=True)

    st.markdown(f'<div class="status-bar">Found <strong>{len(results)}</strong> results for '
                f'<em>"{query}"</em> — indexing {len(news_corpus)} live articles</div>',
                unsafe_allow_html=True)

    if not results:
        st.markdown('<div class="no-results">No stories found. Try enabling Semantic Expansion or a broader term.</div>',
                    unsafe_allow_html=True)
    else:
        cards_html = '<div class="news-grid">'
        for i, (did, score) in enumerate(results):
            art = news_corpus[did]
            cards_html += card_html(art, score=score, hero=(i == 0))
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

else:
    # ── Homepage: show latest articles with hero
    st.markdown(f'<div class="status-bar">Today\'s dispatch — {len(news_corpus)} live articles indexed</div>',
                unsafe_allow_html=True)

    sample = list(news_corpus.values())[:12]
    cards_html = '<div class="news-grid">'
    for i, art in enumerate(sample):
        cards_html += card_html(art, hero=(i == 0))
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
