# 📰 Smart News Information Storage & Retrieval System

A comprehensive Information Retrieval (IR) system built with Python that fetches **live breaking news** from around the globe and lets you search through it using a powerful **TF-IDF Vector Space Model** with optional **WordNet Semantic Expansion**.

---

## 🌟 Features

- **Live News Feed** — Automatically fetches 150+ real-time articles from Google News RSS across World, Business, Technology, Nation, and more
- **TF-IDF Vector Search** — Ranks articles by relevance using the classic Term Frequency–Inverse Document Frequency algorithm
- **Cosine Similarity Ranking** — Results are ranked from most to least relevant using cosine similarity scores
- **WordNet Semantic Expansion** — Optionally expands your query with synonyms from the NLTK WordNet corpus to improve recall
- **Boolean Search Engine** — Classic AND/OR/NOT boolean query engine built on an inverted index
- **Premium Editorial UI** — Beautiful Streamlit interface styled with Playfair Display serif typography and a cream editorial aesthetic
- **Interactive Card Grid** — Search results displayed in a 3-column masonry card layout with clickable links to original articles

---

## 🧠 System Architecture

```
User Query
    │
    ▼
Preprocessing Pipeline
(lowercase → tokenize → stopword removal → lemmatization)
    │
    ├──► Boolean Search Engine (Inverted Index)
    │
    ├──► TF-IDF Vector Space Model + Cosine Similarity
    │
    └──► WordNet Semantic Query Expansion ──► TF-IDF Search
                                                    │
                                                    ▼
                                          Ranked News Results
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| UI & Server | Streamlit |
| NLP Pipeline | NLTK (tokenize, stopwords, lemmatize, WordNet) |
| Information Retrieval | Scikit-learn (TF-IDF, Cosine Similarity) |
| Live Data | Google News RSS (via `requests` + `xml`) |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/rezmi-20/Smart-news-retrival.git
cd Smart-news-retrival
```

### 2. Install Dependencies

```bash
pip install streamlit nltk scikit-learn requests numpy
```

### 3. Run the App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
├── app.py               # Main Streamlit application (Phases 1–6 integrated)
├── test_phase1.py       # Phase 1: Environment setup & corpus definition
├── test_phase2.py       # Phase 2: Text cleaning & NLP preprocessing pipeline
├── test_phase3.py       # Phase 3: Inverted index & Boolean query engine
├── test_phase4.py       # Phase 4: TF-IDF vectorizer & cosine similarity ranking
├── test_phase5.py       # Phase 5: WordNet semantic query expansion
└── README.md
```

---

## 📐 IR Phases Implemented

| Phase | Module | Description |
|---|---|---|
| 1 | Corpus Ingestion | Fetches live news corpus from Google News RSS |
| 2 | Text Preprocessing | Lowercase, regex clean, tokenize, stopword remove, lemmatize |
| 3 | Boolean Engine | Inverted index supporting AND, OR, NOT queries |
| 4 | Vector Space Model | TF-IDF matrix + Cosine similarity ranking |
| 5 | Semantic Expansion | WordNet synset-based query expansion for improved recall |
| 6 | Evaluation | Precision, Recall, and F1-Score metrics |

---

## 📊 Evaluation Metrics

The system's retrieval quality is evaluated using:

- **Precision** = Retrieved Relevant / Total Retrieved
- **Recall** = Retrieved Relevant / Total Relevant
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)

---

## 🎓 Academic Context

This project was developed as a comprehensive assignment for an **Information Retrieval** course, demonstrating the full pipeline from raw text ingestion through ranked retrieval and evaluation — all applied to a real-world live news dataset.

---

## 👤 Author

**Remedan** — [@rezmi-20](https://github.com/rezmi-20)
