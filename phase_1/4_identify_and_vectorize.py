import pandas as pd
import re
import os
import unicodedata
from bs4 import BeautifulSoup
import spacy
import argparse
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def group_news_by_3_days():
    df = pd.read_csv('datasets/all_news.csv', parse_dates=['date'])
    df = df.sort_values(["date"])
    df["full_text"] = df["headline"].fillna('') + " " + df["article"].fillna('')
    aggregated_list = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        for i in range(len(group)):
            start_idx = max(0, i - 2)
            subset = group.loc[start_idx:i, "full_text"]
            aggregated_text = " ".join(subset)
            # Ensure we store only the date part in ISO format (YYYY-MM-DD)
            try:
                the_date = pd.to_datetime(group.loc[i, "date"]).date().isoformat()
            except Exception:
                # fallback: coerce to string and strip
                the_date = str(group.loc[i, "date"]).strip()
            aggregated_list.append({
                "date": the_date,
                "symbol": symbol,
                "news": aggregated_text
            })

    aggregated_df = pd.DataFrame(aggregated_list)
    aggregated_df.to_csv("datasets/aggregated_news.csv", index=False)
    
def preprocess_news():
    
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")

    df = pd.read_csv(src)

    # Helper: strip HTML to visible text
    def extract_text(html_or_text: str) -> str:
        if not isinstance(html_or_text, str):
            return ''
        # quick check for HTML
        if '<' in html_or_text and '>' in html_or_text:
            try:
                return BeautifulSoup(html_or_text, 'html.parser').get_text(separator=' ')
            except Exception:
                return re.sub(r'<[^>]+>', ' ', html_or_text)
        return html_or_text

    # Normalize and clean basic noise
    URL_RE = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_RE = re.compile(r'\S+@\S+')

    def normalize(text: str) -> str:
        if not text:
            return ''
        text = unicodedata.normalize('NFKC', text)
        text = URL_RE.sub(' ', text)
        text = EMAIL_RE.sub(' ', text)
        # remove stray control chars
        text = re.sub(r'[\r\t\x0b\x0c]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Arg parse is handled in __main__; allow callers to set via env/args later
    # Try to load spaCy but do not require the parser by default (heuristic mode uses tokenizer/lemmatizer only)
    try:
        # load model without parser by default; parser is needed for dependency-based negation
        nlp = spacy.load('en_core_web_sm', disable=['parser'])
        has_spacy = True
    except Exception:
        nlp = None
        has_spacy = False

    # Simple negation set
    NEGATIONS = set(['no', 'not', "n't", 'never', 'none', 'nothing', 'neither', 'nor'])
    NEGATION_SCOPE = 3  # mark up to this many content tokens after a negation 

    cleaned_texts = []

    # Decide negation strategy: default 'heuristic'. If caller set env/arg to 'dependency' we will try to use parser-based method.
    negation_mode = getattr(preprocess_news, '_negation_mode', 'heuristic')

    if negation_mode == 'dependency' and has_spacy:
        # We need the parser to find dependency relations for negation. Try to enable it.
        try:
            nlp_parser = spacy.load('en_core_web_sm')
            use_dependency = True
        except Exception:
            # Fall back to heuristic if parser not available
            print("[warn] spaCy parser not available; falling back to heuristic negation", file=sys.stderr)
            nlp_parser = None
            use_dependency = False
    else:
        use_dependency = False

    if use_dependency and nlp_parser is not None:
        # Dependency-based negation: mark tokens that depend on negation words via a limited traversal.
        texts = (extract_text(x) for x in df['news'].astype(str).tolist())
        norm_texts = [normalize(t) for t in texts]
        for doc in nlp_parser.pipe(norm_texts, batch_size=50):
            out_tokens = []
            # Identify negation tokens in the doc (by lemma or text)
            neg_indices = set([tok.i for tok in doc if tok.lemma_.lower() in NEGATIONS or tok.text.lower() in NEGATIONS])
            # For performance, build a mapping of token -> whether it's negated
            negated = [False] * len(doc)
            for ni in neg_indices:
                # Mark children of the negation token as negated (shallow) and token head and siblings
                for child in doc[ni].children:
                    negated[child.i] = True
                # also mark the head (word being modified) if within sentence
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
        # Heuristic: use tokenizer/lemmatizer if we have spaCy (without parser), else fallback to regex tokens
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
                        # punctuation ends any active negation scope
                        neg_remaining = 0
                        continue
                    tok_lower = token.text.lower()
                    if tok_lower in NEGATIONS:
                        # start negation scope for the next content tokens (do not mark the negation token itself)
                        neg_remaining = NEGATION_SCOPE
                        continue
                    if token.is_stop:
                        # skip stopwords entirely
                        continue
                    lemma = token.lemma_.lower()
                    if neg_remaining > 0:
                        out_tokens.append('NOT_' + lemma)
                        neg_remaining -= 1
                    else:
                        out_tokens.append(lemma)
                cleaned_texts.append(' '.join(out_tokens))
        else:
            # Fallback: naive tokenization when spaCy is not available
            for raw in df['news'].astype(str).tolist():
                t = normalize(extract_text(raw))
                toks = re.findall(r"\b\w+\b", t.lower())
                out = []
                neg_remaining = 0
                for w in toks:
                    if w in NEGATIONS:
                        neg_remaining = NEGATION_SCOPE
                        continue
                    # fallback: do not remove stopwords here; mark negated content tokens
                    if neg_remaining > 0:
                        out.append('NOT_' + w)
                        neg_remaining -= 1
                    else:
                        out.append(w)
                cleaned_texts.append(' '.join(out))

    # Update dataframe and overwrite CSV; sort by date descending (most recent first)
    df['news'] = cleaned_texts
    # If there is a date column, try to parse and sort; otherwise skip sorting
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
        except Exception:
            pass
    df.to_csv(src, index=False)
    print(f"Preprocessing complete: updated {src} ({len(df)} rows)")
    
    
def vectorize_dtm():
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)
    if 'symbol' not in df.columns:
        raise KeyError(f"Input CSV {src} missing required column 'symbol'. Columns: {list(df.columns)}")
    vectorizer = CountVectorizer(max_features=5000, min_df=2)
    dtm = vectorizer.fit_transform(df['news'].astype(str).tolist())
    # Build a dense vector representation per document (as list) and store as a single column
    vectors = dtm.toarray().tolist()

    out = pd.DataFrame({
        'symbol': df['symbol'],
        'date': df['date'],
        # store the raw vector as a stringified list so it matches expected sample format
        'news_vector': [str(v) for v in vectors]
    })
    # attach impact scores (if available) and write CSV
    out = merge_impact_scores(out)
    out = out[['symbol', 'date', 'news_vector', 'impact_score']]
    dtm_src = 'datasets/vectorized_news_dtm.csv'
    out.to_csv(dtm_src, index=False)
    print(f"Document-term matrix saved to {dtm_src} ({out.shape[0]} rows, columns={list(out.columns)})")
def merge_impact_scores(out_df, impact_candidates=None):
    """
    Normalize keys and merge impact_score from the first available impact CSV.

    Parameters
    - out_df: DataFrame with at least 'symbol' and 'date' columns
    - impact_candidates: optional list of file paths to consider (ordered)

    Returns merged DataFrame with an 'impact_score' column (may be NA if no file found)
    """
    if impact_candidates is None:
        impact_candidates = ['datasets/historical_prices_impact.csv', 'datasets/sample_historical_prices_impact.csv']

    out = out_df.copy()
    # normalize out keys
    out['symbol'] = out['symbol'].astype(str).str.strip().str.upper()
    # try parsing date; if there are timezone offsets or timestamps, try to extract date portion
    out_dates = pd.to_datetime(out['date'], errors='coerce')
    # For entries that failed to parse, try to extract an ISO-like date substring via regex
    if out_dates.isna().any():
        import re as _re
        def _extract_date(s):
            if not isinstance(s, str):
                return s
            m = _re.search(r"(\d{4}-\d{2}-\d{2})", s)
            if m:
                return m.group(1)
            m2 = _re.search(r"(\d{2}/\d{2}/\d{4})", s)
            if m2:
                # convert dd/mm/YYYY or mm/dd/YYYY ambiguous; leave for pandas to coerce
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

            # normalize impact keys
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
            # coalesce in case of duplicate columns
            if 'impact_score_x' in out.columns and 'impact_score_y' in out.columns:
                out['impact_score'] = out['impact_score_y'].fillna(out['impact_score_x'])
                out.drop(['impact_score_x', 'impact_score_y'], axis=1, inplace=True)

            print(f"Merged impact scores from {p} into DTM output")
            return out

    # no impact file found -> create empty column and warn
    out['impact_score'] = pd.NA
    print("No impact CSV found; `impact_score` will be empty. Placeholders emitted for later merging.")
    return out

def vectorize_tfidf():
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)
    if 'symbol' not in df.columns:
        raise KeyError(f"Input CSV {src} missing required column 'symbol'. Columns: {list(df.columns)}")
    vec = TfidfVectorizer(max_features=5000, min_df=2)
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
    Curated feature matrix: counts (or presence) of chosen sentiment-bearing words.
    """
    src = 'datasets/aggregated_news.csv'
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing aggregated news CSV: {src}")
    df = pd.read_csv(src)

    # curated list of sentiment-bearing words (example from assignment)
    ###------CHANGE THESE WORDS AS NEEDED------###
    curated_words = [
        'gain', 'loss', 'strong', 'weak', 'upgrade', 'downgrade', 'beat', 'miss', 'bullish', 'bearish'
    ]
    # Use CountVectorizer with fixed vocabulary to get counts for these terms
    cv = CountVectorizer(vocabulary={w: i for i, w in enumerate(curated_words)}, lowercase=True)
    # With a fixed vocabulary we can directly transform
    mat = cv.transform(df['news'].astype(str).tolist())
    vectors = mat.toarray().tolist()

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

    # (no duplicated writes here — each vectorizer writes its own CSV earlier)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate and preprocess news for sentiment tasks')
    parser.add_argument('--negation-mode', choices=['heuristic', 'dependency'], default='heuristic',
                        help='Negation handling strategy: heuristic (short scope) or dependency (spaCy parser-based)')
    args = parser.parse_args()

    # Attach chosen negation mode to function so it can be read inside
    preprocess_news._negation_mode = args.negation_mode

    print(1)
    group_news_by_3_days()
    print(2)
    preprocess_news()
    print(3)
    vectorize_dtm()
    print(4)
    vectorize_tfidf()
    print(5)
    vectorize_curated()
