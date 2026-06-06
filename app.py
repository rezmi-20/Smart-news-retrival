import streamlit as st
import html as html_lib
import re
import numpy as np
import requests
import xml.etree.ElementTree as ET
import nltk
from datetime import datetime
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="The News Dispatch", page_icon="📰", layout="wide",
                   initial_sidebar_state="collapsed")

# ── CSS  (only touches non-widget elements + overrides that are reliable) ────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem !important; max-width: 1200px !important; margin: auto !important; }

/* masthead */
.masthead { text-align:center; border-bottom: 3px double #111; padding-bottom:16px; margin-bottom:0; }
.masthead-date { font-family:'Inter',sans-serif; font-size:.75rem; letter-spacing:2px; text-transform:uppercase; color:#888; }
.masthead-logo { font-family:'Playfair Display',serif; font-size:clamp(2.2rem,5vw,4rem); color:#111; margin:4px 0; line-height:1; }
.masthead-sub { font-family:'Inter',sans-serif; font-size:.75rem; color:#888; letter-spacing:1px; }

/* divider */
.divider { border:none; border-top:1px solid #ccc; margin:8px 0; }

/* status */
.status { font-family:'Inter',sans-serif; font-size:.8rem; color:#666; margin:6px 0 12px; }

/* cards grid */
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:20px; margin-top:16px; }

/* individual card */
.ncard { background:#fff; border:1px solid #e5e0d8; border-radius:6px; overflow:hidden;
         display:flex; flex-direction:column; transition:transform .2s,box-shadow .2s; }
.ncard:hover { transform:translateY(-5px); box-shadow:0 10px 30px rgba(0,0,0,.12); }
.ncard img { width:100%; aspect-ratio:16/9; object-fit:cover; display:block; }
.ncard-body { padding:14px 16px 16px; display:flex; flex-direction:column; gap:6px; flex:1; }
.ncard-cat { font-family:'Inter',sans-serif; font-size:.65rem; font-weight:700;
             letter-spacing:1.5px; text-transform:uppercase; color:#c0392b; }
.ncard-title { font-family:'Playfair Display',serif; font-size:1.05rem; line-height:1.3;
               color:#111; margin:0; }
.ncard-link { margin-top:auto; padding-top:10px; font-family:'Inter',sans-serif;
              font-size:.75rem; font-weight:600; color:#c0392b; text-decoration:none; }
.ncard-score { font-family:'Inter',sans-serif; font-size:.7rem; color:#999; font-style:italic; }

/* hero card: first result spans 2 cols */
.hero { grid-column: span 2; }
.hero img { aspect-ratio:21/9; }
.hero .ncard-title { font-size:1.6rem; }

/* semantic pill */
.sem-pill { background:#fff3f3; border-left:3px solid #c0392b; padding:8px 14px;
            font-family:'Inter',sans-serif; font-size:.8rem; color:#555;
            margin-bottom:10px; border-radius:0 4px 4px 0; }

/* no results */
.no-res { text-align:center; padding:60px 20px;
          font-family:'Playfair Display',serif; font-size:1.5rem;
          font-style:italic; color:#aaa; }
</style>
""", unsafe_allow_html=True)

# ── NLTK ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def download_nltk():
    for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab']:
        nltk.download(p, quiet=True)
download_nltk()

# ── Image generation from article title keywords ─────────────────────────────
# Unsplash Source API: returns a relevant photo for given comma-separated keywords.
# No API key required. Falls back to a category keyword if title yields nothing.
CAT_FALLBACK = {
    'WORLD':      'global,world,politics',
    'NATION':     'government,nation,policy',
    'BUSINESS':   'finance,business,economy',
    'TECHNOLOGY': 'technology,digital,innovation',
    'SPORTS':     'sports,athlete,game',
    'HEALTH':     'health,medicine,hospital',
    'LATEST':     'news,newspaper,journalism',
}

_img_stop = {
    'the','and','for','are','but','not','you','all','any','can','had',
    'her','was','one','our','out','day','get','has','him','his','how',
    'man','new','now','old','see','two','way','who','did','its','let',
    'put','say','she','too','use','that','this','with','from','they',
    'will','been','have','more','over','such','than','then','them',
    'well','were','what','when','also','into','just','like','make',
    'most','some','time','very','after','could','first','their','there',
    'which','would','about','other','these','those','being','since',
    'amid','says','said','amid','amid','amid',
}

def get_img_for_article(title: str, category: str) -> str:
    """Build a relevant image URL from the article title's key nouns via loremflickr."""
    words = [
        w for w in re.sub(r'[^a-z\s]', '', title.lower()).split()
        if w not in _img_stop and len(w) > 3
    ]
    if words:
        keywords = ','.join(words[:3])
    else:
        keywords = CAT_FALLBACK.get(category, 'news')
    # loremflickr.com — free, active, returns relevant Flickr photos by keyword
    return f'https://loremflickr.com/800/450/{keywords}'

# ── Fetch live news ───────────────────────────────────────────────────────────
FEEDS = [
    ('LATEST',      'https://news.google.com/rss'),
    ('WORLD',       'https://news.google.com/rss/headlines/section/topic/WORLD'),
    ('NATION',      'https://news.google.com/rss/headlines/section/topic/NATION'),
    ('BUSINESS',    'https://news.google.com/rss/headlines/section/topic/BUSINESS'),
    ('TECHNOLOGY',  'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY'),
    ('SPORTS',      'https://news.google.com/rss/headlines/section/topic/SPORTS'),
    ('HEALTH',      'https://news.google.com/rss/headlines/section/topic/HEALTH'),
]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_news():
    corpus = {}
    doc_id = 1
    for cat, url in FEEDS:
        try:
            res = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            root = ET.fromstring(res.content)
            for item in root.findall('.//item')[:25]:
                t = item.find('title')
                l = item.find('link')
                if t is None or l is None: continue
                raw = t.text or ''
                title = re.sub(r'\s*-\s*[^-]+$', '', raw).strip()
                corpus[doc_id] = {'title': title, 'link': l.text,
                                  'category': cat,
                                  'image': get_img_for_article(title, cat)}
                doc_id += 1
        except Exception:
            pass
    return corpus

# ── NLP pipeline ─────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words('english')) | {'vs','say','said','says','uk','mr','new'}

def clean(text):
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    return [lemmatizer.lemmatize(w) for w in word_tokenize(text)
            if w not in STOP and len(w) > 2]

@st.cache_data(show_spinner=False)
def build_index(keys, vals):
    doc_ids, texts = [], []
    for did, art in zip(keys, vals):
        toks = clean(art['title']) + clean(art['category'])
        doc_ids.append(did)
        texts.append(" ".join(toks))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    return doc_ids, vec, mat

def tfidf_search(query, corpus, doc_ids, vec, mat, cat_filter='ALL', top_k=12):
    pq = " ".join(clean(query))
    if not pq: return []
    sims = cosine_similarity(vec.transform([pq]), mat).flatten()
    ranked = np.argsort(sims)[::-1]
    out = []
    for i in ranked:
        if sims[i] < 0.001: break
        did = doc_ids[i]
        art = corpus[did]
        if cat_filter != 'ALL' and art['category'] != cat_filter:
            continue
        out.append((did, float(sims[i])))
        if len(out) >= top_k: break
    return out

def semantic_expand(query):
    toks = clean(query)
    exp = set(toks)
    for tok in toks:
        for syn in list(wordnet.synsets(tok))[:2]:
            for lemma in list(syn.lemmas())[:3]:
                exp.update(clean(lemma.name().replace('_',' ')))
    return " ".join(exp)

# ── Load data ─────────────────────────────────────────────────────────────
with st.spinner("Fetching today's dispatch..."):
    corpus = fetch_live_news()

if not corpus:
    st.error("Could not fetch news. Check your internet connection.")
    st.stop()

# ── Sidebar refresh ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### The News Dispatch")
    st.caption(f"{len(corpus)} articles indexed")
    if st.button("Refresh News Feed", use_container_width=True):
        fetch_live_news.clear()
        st.rerun()

keys  = tuple(corpus.keys())
vals  = tuple(corpus.values())
doc_ids, tfidf_vec, tfidf_mat = build_index(keys, vals)

# ── Session state ─────────────────────────────────────────────────────────────
if 'category' not in st.session_state:
    st.session_state.category = 'ALL'
if 'query' not in st.session_state:
    st.session_state.query = ''
if 'semantic' not in st.session_state:
    st.session_state.semantic = False

# ── MASTHEAD ──────────────────────────────────────────────────────────────────
today = datetime.now().strftime("%A, %B %d, %Y").upper()
st.markdown(f"""
<div class="masthead">
  <div class="masthead-date">{today}</div>
  <div class="masthead-logo">The News Dispatch</div>
  <div class="masthead-sub">Live &middot; Intelligent &middot; Searchable &nbsp;|&nbsp; {len(corpus)} articles indexed</div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ── SEARCH FORM ───────────────────────────────────────────────────────────────
with st.form("search_form"):
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        q_input = st.text_input("Search", value=st.session_state.query,
                                placeholder="Search news — e.g. technology, iran, economy ...",
                                label_visibility="collapsed")
    with col_btn:
        submitted = st.form_submit_button("Search", use_container_width=True)
    use_sem = st.checkbox("Enable Semantic WordNet Expansion", value=st.session_state.semantic)

if submitted:
    st.session_state.query    = q_input.strip()
    st.session_state.semantic = use_sem
    st.session_state.category = 'ALL'   # reset category on new search

# ── CATEGORY FILTER ───────────────────────────────────────────────────────────
cats = ['ALL','WORLD','NATION','BUSINESS','TECHNOLOGY','SPORTS','HEALTH']
cat_cols = st.columns(len(cats))
for i, cat in enumerate(cats):
    with cat_cols[i]:
        label = f"**{cat}**" if cat == st.session_state.category else cat
        if st.button(label, key=f"cat_{cat}", use_container_width=True):
            st.session_state.category = cat

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── SEARCH & RENDER ───────────────────────────────────────────────────────────
active_query = st.session_state.query
active_cat   = st.session_state.category

def card_html(art, score=None, hero=False):
    cls        = "ncard hero" if hero else "ncard"
    safe_title = html_lib.escape(art['title'])
    safe_cat   = html_lib.escape(art['category'])
    safe_link  = art['link'].replace('"', '%22')
    safe_img   = art['image'].replace('"', '%22')
    score_line = f'<div class="ncard-score">Relevance: {score:.3f}</div>' if score else ''
    return (
        f'<a class="{cls}" href="{safe_link}" target="_blank" style="text-decoration:none;">'
        f'<img src="{safe_img}" loading="lazy" '
        f'onerror="this.src=\'https://images.unsplash.com/photo-1504711434969-e33886168f5c?w=800&q=80\'">'
        f'<div class="ncard-body">'
        f'<span class="ncard-cat">{safe_cat}</span>'
        f'<h3 class="ncard-title">{safe_title}</h3>'
        f'{score_line}'
        f'<span class="ncard-link">Read full story \u2192</span>'
        f'</div></a>'
    )

if active_query:
    # Run search
    expand_note = ""
    if st.session_state.semantic:
        expanded = semantic_expand(active_query)
        results  = tfidf_search(expanded, corpus, doc_ids, tfidf_vec, tfidf_mat, active_cat)
        expand_note = f'<div class="sem-pill">Semantic expansion: <em>{expanded}</em></div>'
    else:
        results  = tfidf_search(active_query, corpus, doc_ids, tfidf_vec, tfidf_mat, active_cat)

    if expand_note:
        st.markdown(expand_note, unsafe_allow_html=True)

    st.markdown(f'<div class="status">Found <strong>{len(results)}</strong> results for '
                f'<em>"{active_query}"</em>'
                + (f' in <strong>{active_cat}</strong>' if active_cat != 'ALL' else '') +
                '</div>', unsafe_allow_html=True)

    if not results:
        st.markdown('<div class="no-res">No stories found — try Semantic Expansion or a broader term.</div>',
                    unsafe_allow_html=True)
    else:
        html = '<div class="grid">'
        for i, (did, score) in enumerate(results):
            html += card_html(corpus[did], score=score, hero=(i == 0))
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

else:
    # Homepage: show latest by selected category
    if active_cat == 'ALL':
        sample = list(corpus.values())[:12]
    else:
        sample = [a for a in corpus.values() if a['category'] == active_cat][:12]

    st.markdown(f'<div class="status">Showing latest'
                + (f' <strong>{active_cat}</strong>' if active_cat != 'ALL' else '') +
                f' stories — {len(corpus)} articles indexed</div>', unsafe_allow_html=True)

    html = '<div class="grid">'
    for i, art in enumerate(sample):
        html += card_html(art, hero=(i == 0))
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
