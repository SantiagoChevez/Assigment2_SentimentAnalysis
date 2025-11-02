import pandas as pd
import re
import os
import unicodedata
from bs4 import BeautifulSoup
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter

NEGATION_MODE = 'dependency'  # Options: 'dependency' (requires spaCy parser) or 'heuristic'



def group_news_by_3_days():
    """
    Aggregate news articles by symbol into 3-day rolling windows.

    For each symbol and date, aggregates news from the current day and the previous 2 days.
    This creates temporal context for sentiment analysis. Text length is limited to prevent
    memory issues during processing.

    Input:  datasets/all_news.csv
    Output: datasets/aggregated_news.csv

    Raises:
    -------
    FileNotFoundError
        If all_news.csv is not found.
    """
    MAX_ARTICLE_LEN = 100000
    MAX_TEXT_LEN = 200000
    MAX_AGGREGATED_LEN = 500000

    df = pd.read_csv('datasets/all_news.csv', parse_dates=['date'])
    df = df.sort_values(["date"])

    df["article_limited"] = df["article"].fillna('').astype(str).str[:MAX_ARTICLE_LEN]
    df["full_text"] = df["headline"].fillna('') + " " + df["article_limited"]
    aggregated_list = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        for i in range(len(group)):
            start_idx = max(0, i - 2)
            subset = group.loc[start_idx:i, "full_text"]
            limited_texts = [str(text)[:MAX_TEXT_LEN] if len(str(text)) > MAX_TEXT_LEN else str(text) for text in subset]
            aggregated_text = " ".join(limited_texts)
            if len(aggregated_text) > MAX_AGGREGATED_LEN:
                aggregated_text = aggregated_text[:MAX_AGGREGATED_LEN]

            try:
                the_date = pd.to_datetime(group.loc[i, "date"]).date().isoformat()
            except Exception:
                the_date = str(group.loc[i, "date"]).strip()

            aggregated_list.append({
                "date": the_date,
                "symbol": symbol,
                "news": aggregated_text
            })

    aggregated_df = pd.DataFrame(aggregated_list)
    aggregated_df.to_csv("datasets/aggregated_news.csv", index=False)


def preprocess_news():
    """
    Preprocess news text: remove HTML, normalize, handle negation.

    Performs comprehensive text preprocessing:
    - Strips HTML tags and extracts visible text
    - Normalizes unicode, removes URLs and emails
    - Handles negation using dependency parsing (if available) or heuristic method
    - Lemmatizes tokens and removes stopwords
    - Marks negated tokens with 'NOT_' prefix

    Input:  datasets/aggregated_news.csv
    Output: datasets/aggregated_news.csv (overwritten with preprocessed text)

    Raises:
    -------
    FileNotFoundError
        If aggregated_news.csv is not found.
    """
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")

    df = pd.read_csv(src)

    def extract_text(html_or_text: str) -> str:
        """Extract visible text from HTML or return plain text."""
        if not isinstance(html_or_text, str):
            return ''
        if '<' in html_or_text and '>' in html_or_text:
            try:
                return BeautifulSoup(html_or_text, 'html.parser').get_text(separator=' ')
            except Exception:
                return re.sub(r'<[^>]+>', ' ', html_or_text)
        return html_or_text

    URL_RE = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_RE = re.compile(r'\S+@\S+')

    def normalize(text: str) -> str:
        """Normalize text: unicode, remove URLs/emails, clean whitespace."""
        if not text:
            return ''
        text = unicodedata.normalize('NFKC', text)
        text = URL_RE.sub(' ', text)
        text = EMAIL_RE.sub(' ', text)
        text = re.sub(r'[\r\t\x0b\x0c]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser'])
        has_spacy = True
    except Exception:
        nlp = None
        has_spacy = False

    NEGATIONS = set(['no', 'not', "n't", 'never', 'none', 'nothing', 'neither', 'nor'])
    NEGATION_SCOPE = 3

    cleaned_texts = []
    negation_mode = NEGATION_MODE

    if negation_mode == 'dependency' and has_spacy:
        try:
            nlp_parser = spacy.load('en_core_web_sm')
            use_dependency = True
        except Exception:
            print("[warn] spaCy parser not available; falling back to heuristic negation", file=sys.stderr)
            nlp_parser = None
            use_dependency = False
    else:
        use_dependency = False

    if use_dependency and nlp_parser is not None:
        # Dependency-based negation: use syntactic dependencies to identify negated tokens
        texts = (extract_text(x) for x in df['news'].astype(str).tolist())
        norm_texts = [normalize(t) for t in texts]
        for doc in nlp_parser.pipe(norm_texts, batch_size=50):
            out_tokens = []
            neg_indices = set([tok.i for tok in doc if tok.lemma_.lower() in NEGATIONS or tok.text.lower() in NEGATIONS])
            negated = [False] * len(doc)
            for ni in neg_indices:
                for child in doc[ni].children:
                    negated[child.i] = True
                head = doc[ni].head
                if head is not None:
                    negated[head.i] = True
            for token in doc:
                if token.is_space or token.is_punct or token.is_stop:
                    continue
                lemma = token.lemma_.lower()
                if negated[token.i]:
                    out_tokens.append('NOT_' + lemma)
                else:
                    out_tokens.append(lemma)
            cleaned_texts.append(' '.join(out_tokens))
    else:
        # Heuristic negation: mark N tokens after negation words
        if has_spacy and nlp is not None:
            texts = (extract_text(x) for x in df['news'].astype(str).tolist())
            norm_texts = [normalize(t) for t in texts]
            for doc in nlp.pipe(norm_texts, batch_size=50):
                out_tokens = []
                neg_remaining = 0
                for token in doc:
                    if token.is_space:
                        continue
                    if token.is_punct:
                        neg_remaining = 0
                        continue
                    tok_lower = token.text.lower()
                    if tok_lower in NEGATIONS:
                        neg_remaining = NEGATION_SCOPE
                        continue
                    if token.is_stop:
                        continue
                    lemma = token.lemma_.lower()
                    if neg_remaining > 0:
                        out_tokens.append('NOT_' + lemma)
                        neg_remaining -= 1
                    else:
                        out_tokens.append(lemma)
                cleaned_texts.append(' '.join(out_tokens))
        else:
            # Fallback: simple regex tokenization without spaCy
            for raw in df['news'].astype(str).tolist():
                t = normalize(extract_text(raw))
                toks = re.findall(r"\b\w+\b", t.lower())
                out = []
                neg_remaining = 0
                for w in toks:
                    if w in NEGATIONS:
                        neg_remaining = NEGATION_SCOPE
                        continue
                    if neg_remaining > 0:
                        out.append('NOT_' + w)
                        neg_remaining -= 1
                    else:
                        out.append(w)
                cleaned_texts.append(' '.join(out))

    df['news'] = cleaned_texts
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        except Exception:
            pass
    df.to_csv(src, index=False)
    print(f"Preprocessing complete: updated {src} ({len(df)} rows)")

def merge_impact_scores(out_df, impact_candidates=None):
    """
    Merge impact scores from historical prices impact file into vectorized news DataFrame.

    Normalizes symbol and date columns for matching, then performs left join on (symbol, date).
    Tries multiple candidate files in order until one is found.

    Parameters:
    -----------
    out_df : pd.DataFrame
        DataFrame with at least 'symbol' and 'date' columns to merge scores into.
    impact_candidates : list of str, optional
        Ordered list of CSV file paths to try. Defaults to standard impact file paths.

    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'impact_score' column. Values may be NA if no matching file found.
    """
    if impact_candidates is None:
        impact_candidates = ['datasets/historical_prices_impact.csv']

    out = out_df.copy()
    out['symbol'] = out['symbol'].astype(str).str.strip().str.upper()
    out_dates = pd.to_datetime(out['date'], errors='coerce')
    if out_dates.isna().any():
        import re as _re
        def _extract_date(s):
            """Extract date from string using regex patterns."""
            if not isinstance(s, str):
                return s
            m = _re.search(r"(\d{4}-\d{2}-\d{2})", s)
            if m:
                return m.group(1)
            m2 = _re.search(r"(\d{2}/\d{2}/\d{4})", s)
            if m2:
                return m2.group(1)
            return s
        out['date'] = out['date'].astype(str).apply(_extract_date)
        try:
            out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        except Exception:
            out['date'] = out['date'].astype(str).str.strip()
    else:
        out['date'] = out_dates.dt.strftime('%Y-%m-%d')

    for p in impact_candidates:
        if os.path.exists(p):
            try:
                imp = pd.read_csv(p)
            except Exception:
                imp = pd.read_csv(p, dtype=str, low_memory=False)

            if 'symbol' in imp.columns:
                imp['symbol'] = imp['symbol'].astype(str).str.strip().str.upper()
            if 'date' in imp.columns:
                try:
                    imp['date'] = pd.to_datetime(imp['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                except Exception:
                    imp['date'] = imp['date'].astype(str).str.strip()

            if 'impact_score' not in imp.columns:
                imp['impact_score'] = pd.NA

            out = out.merge(imp[['symbol', 'date', 'impact_score']], on=['symbol', 'date'], how='left')
            if 'impact_score_x' in out.columns and 'impact_score_y' in out.columns:
                out['impact_score'] = out['impact_score_y'].fillna(out['impact_score_x'])
                out.drop(['impact_score_x', 'impact_score_y'], axis=1, inplace=True)

            print(f"Merged impact scores from {p}")
            return out

    out['impact_score'] = pd.NA
    print("No impact CSV found; `impact_score` will be empty. Placeholders emitted for later merging.")
    return out


def vectorize_dtm():
    """
    Create Document-Term Matrix (DTM) vectorization of news text.

    Uses CountVectorizer to create bag-of-words features. Automatically reduces
    feature count for large datasets to manage memory. Merges impact scores
    and saves to CSV.

    Input:  datasets/aggregated_news.csv
    Output: datasets/vectorized_news_dtm.csv

    Raises:
    -------
    FileNotFoundError
        If aggregated_news.csv is not found.
    KeyError
        If required columns are missing.
    """
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)
    if 'symbol' not in df.columns:
        raise KeyError(f"Input CSV {src} missing required column 'symbol'. Columns: {list(df.columns)}")

    max_features = 5000
    min_df = 2
    if len(df) > 100_000:
        print(f"[WARN] Large dataset detected ({len(df)} rows). Reducing max_features to 1000 for DTM.")
        max_features = 1000
        min_df = 5

    vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
    dtm = vectorizer.fit_transform(df['news'].astype(str).tolist())
    vectors = dtm.toarray().tolist()

    out = pd.DataFrame({
        'symbol': df['symbol'],
        'date': df['date'],
        'news_vector': [str(v) for v in vectors]
    })
    out = merge_impact_scores(out)
    out = out[['symbol', 'date', 'news_vector', 'impact_score']]
    dtm_src = 'datasets/vectorized_news_dtm.csv'
    out.to_csv(dtm_src, index=False)
    print(f"Document-term matrix saved to {dtm_src} ({out.shape[0]} rows, columns={list(out.columns)})")


def vectorize_tfidf():
    """
    Create TF-IDF vectorization of news text.

    Uses TfidfVectorizer to create term frequency-inverse document frequency features.
    Automatically reduces feature count for large datasets. Merges impact scores
    and saves to CSV.

    Input:  datasets/aggregated_news.csv
    Output: datasets/vectorized_news_tfidf.csv

    Raises:
    -------
    FileNotFoundError
        If aggregated_news.csv is not found.
    KeyError
        If required columns are missing.
    """
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)
    if 'symbol' not in df.columns:
        raise KeyError(f"Input CSV {src} missing required column 'symbol'. Columns: {list(df.columns)}")

    max_features = 5000
    min_df = 2
    if len(df) > 100_000:
        print(f"[WARN] Large dataset detected ({len(df)} rows). Reducing max_features to 1000 for TF-IDF.")
        max_features = 1000
        min_df = 5

    vec = TfidfVectorizer(max_features=max_features, min_df=min_df)
    tfidf = vec.fit_transform(df['news'].astype(str).tolist())
    vectors = tfidf.toarray().tolist()

    out = pd.DataFrame({
        'symbol': df['symbol'],
        'date': df['date'],
        'news_vector': [str(v) for v in vectors]
    })
    out = merge_impact_scores(out)
    out = out[['symbol', 'date', 'news_vector', 'impact_score']]
    dst = 'datasets/vectorized_news_tfidf.csv'
    out.to_csv(dst, index=False)
    print(f"TF-IDF matrix saved to {dst} ({out.shape[0]} rows)")


def vectorize_curated():
    """
    Create curated feature vectorization using sentiment-bearing keywords.

    Counts occurrences of specific sentiment words (and their negated forms) to create
    a 10-dimensional feature vector. Words: buy, sell, beat, miss, guidance, dividend,
    deal, cut, upgrade, plunge. Handles both regular and negated forms (e.g., 'buy' and 'NOT_buy').

    Input:  datasets/aggregated_news.csv
    Output: datasets/vectorized_news_curated.csv

    Raises:
    -------
    FileNotFoundError
        If aggregated_news.csv is not found.
    """
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)

    curated_words = ['buy', 'bullish', 'beat', 'spike', 'profit', 'sell', 'bearish', 'decline', 'weak', 'loss']

    def token_counts(text: str) -> Counter:
        """Count token occurrences in text."""
        toks = str(text).split()
        return Counter(toks)

    vectors = []
    for text in df['news'].astype(str).tolist():
        cnt = token_counts(text)
        vec = []
        for t in curated_words:
            count = cnt.get(t, 0) + cnt.get(f"NOT_{t}", 0)
            vec.append(int(count))
        vectors.append(vec)

    out = pd.DataFrame({
        'symbol': df['symbol'],
        'date': df['date'],
        'news_vector': [str(v) for v in vectors]
    })
    out = merge_impact_scores(out)
    out = out[['symbol', 'date', 'news_vector', 'impact_score']]
    dst = 'datasets/vectorized_news_curated.csv'
    out.to_csv(dst, index=False)
    print(f"Curated feature matrix saved to {dst} ({out.shape[0]} rows)")


if __name__ == "__main__":
    # Execute the full preprocessing and vectorization pipeline:
    # 1) Aggregate news by 3-day windows
    # 2) Preprocess text (HTML removal, normalization, negation handling)
    # 3) Vectorize using curated features
    # 4) Merge impact scores
    print(f"Running pipeline with negation mode: {NEGATION_MODE}")
    group_news_by_3_days()
    print("Aggregated news by 3-day windows.")
    preprocess_news()
    print("Preprocessing complete.")
    print("Starting vectorization...")
    vectorize_dtm()
    print("DTM vectorization complete.")
    vectorize_tfidf()
    print("TF-IDF vectorization complete.")
    vectorize_curated()
    print("Curated vectorization complete.")
    print("Vectorization pipeline complete.")
