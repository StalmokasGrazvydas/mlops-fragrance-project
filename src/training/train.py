import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib



def resolve_input_file(input_path: str) -> str:
    """
    Azure ML Studio sometimes mounts uploaded 'file' inputs as a directory.
    If a directory is provided, pick the first supported file inside it.
    """
    p = Path(input_path)

    if p.is_dir():
        candidates = (
            list(p.glob("*.json"))
            + list(p.glob("*.jsonl"))
            + list(p.glob("*.ndjson"))
            + list(p.glob("*.csv"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No supported data files found in directory: {input_path}. "
                f"Expected one of: .json, .jsonl, .ndjson, .csv"
            )
        # Deterministic choice (sorted)
        candidates = sorted(candidates, key=lambda x: x.name)
        return str(candidates[0])

    return input_path

def rating_to_sentiment(r: float) -> str:
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def load_reviews(path: str, sample_rows: int | None = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    # CSV
    if ext == ".csv":
        df = pd.read_csv(path)
        return df.head(sample_rows) if sample_rows else df

    # JSON / JSONL / NDJSON
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        df = pd.read_json(path)

    return df.head(sample_rows) if sample_rows else df


def ensure_parent_dir(file_path: str) -> None:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def main(input_path: str, model_out: str, metrics_out: str,
         text_col: str, rating_col: str, sample_rows: int | None):
    input_path = resolve_input_file(input_path)
    print(f"Resolved input path: {input_path}")
    df = load_reviews(input_path, sample_rows=sample_rows)

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found. Available columns: {list(df.columns)}")
    if rating_col not in df.columns:
        raise ValueError(f"Rating column '{rating_col}' not found. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[text_col, rating_col]).copy()
    df["sentiment"] = df[rating_col].apply(rating_to_sentiment)

    X = df[text_col].astype(str)
    y = df["sentiment"].astype(str)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_text = classification_report(y_test, preds)

    print(report_text)
    print(f"Accuracy: {acc:.4f}")

    # Ensure output folders exist (critical for Azure ML outputs)
    ensure_parent_dir(model_out)
    ensure_parent_dir(metrics_out)

    joblib.dump(pipeline, model_out)
    print(f"Saved model to: {model_out}")

    metrics_payload = {
        "accuracy": float(acc),
        "classes": sorted(list(set(y_test))),
        "report": report_dict
    }
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics to: {metrics_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to .json/.jsonl/.csv")
    parser.add_argument("--model_out", required=True, help="Output path for model.pkl")
    parser.add_argument("--metrics_out", default="metrics/metrics.json", help="Output path for metrics.json")
    parser.add_argument("--text_col", default="reviewText", help="Common names: reviewText, review_text, text")
    parser.add_argument("--rating_col", default="overall", help="Common names: overall, rating, score, stars")
    parser.add_argument("--sample_rows", type=int, default=200000, help="Subset to speed up training")
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        model_out=args.model_out,
        metrics_out=args.metrics_out,
        text_col=args.text_col,
        rating_col=args.rating_col,
        sample_rows=args.sample_rows
    )
