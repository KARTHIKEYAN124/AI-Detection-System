"""
Dataset-backed training pipeline for OriginScript AI.

The trainer can ingest:
- Kaggle/local CSV or JSONL files placed under data/raw
- Kaggle datasets downloaded with kagglehub when credentials are configured
- HC3 human vs ChatGPT records through the Hugging Face datasets package

Labels use 1 for AI-generated text and 0 for human-written text.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
from urllib.request import urlopen
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion


TEXT_COLUMNS = (
    "text",
    "Text",
    "content",
    "essay",
    "answer",
    "response",
    "full_text",
    "generated_text",
    "human_text",
)
LABEL_COLUMNS = (
    "label",
    "Label",
    "class",
    "target",
    "generated",
    "is_ai",
    "source",
    "writer",
    "model",
)
AI_LABELS = {"1", "ai", "machine", "generated", "chatgpt", "gpt", "llm", "bot", "synthetic"}
HUMAN_LABELS = {"0", "human", "student", "real", "original", "organic", "expert", "person"}


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0 if int(value) == 0 else None
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return int(value)

    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    tokens = set(normalized.split())
    if normalized in AI_LABELS or tokens & AI_LABELS:
        return 1
    if normalized in HUMAN_LABELS or tokens & HUMAN_LABELS:
        return 0
    return None


def infer_text_column(df: pd.DataFrame) -> str | None:
    for column in TEXT_COLUMNS:
        if column in df.columns:
            return column
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if not object_columns:
        return None
    return max(object_columns, key=lambda column: df[column].astype(str).str.len().median())


def infer_label_column(df: pd.DataFrame) -> str | None:
    for column in LABEL_COLUMNS:
        if column in df.columns:
            mapped = df[column].map(normalize_label)
            if mapped.notna().mean() >= 0.8:
                return column
    for column in df.columns:
        mapped = df[column].map(normalize_label)
        if mapped.notna().mean() >= 0.8 and mapped.nunique(dropna=True) == 2:
            return column
    return None


def frame_from_text_label(df: pd.DataFrame, source: str) -> pd.DataFrame:
    text_col = infer_text_column(df)
    label_col = infer_label_column(df)
    if not text_col or not label_col:
        return pd.DataFrame(columns=["text", "label", "source"])

    out = pd.DataFrame(
        {
            "text": df[text_col].map(clean_text),
            "label": df[label_col].map(normalize_label),
            "source": source,
        }
    )
    return out.dropna(subset=["label"])


def read_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_local_files(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() == ".csv":
            raw = pd.read_csv(path)
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            raw = read_jsonl(path)
        elif path.suffix.lower() == ".json":
            raw = pd.read_json(path)
        else:
            continue
        parsed = frame_from_text_label(raw, source=str(path))
        if not parsed.empty:
            frames.append(parsed)
            print(f"Loaded {len(parsed):,} labeled rows from {path}")
        else:
            print(f"Skipped {path}; could not infer text/label columns.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["text", "label", "source"])


def download_kaggle_dataset(slug: str, data_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub to download Kaggle datasets: pip install kagglehub") from exc

    print(f"Downloading Kaggle dataset: {slug}")
    downloaded = Path(kagglehub.dataset_download(slug))
    target = data_dir / "kaggle" / slug.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    for item in downloaded.rglob("*"):
        if item.is_file():
            rel = item.relative_to(downloaded)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
    return target


def flatten_answers(value: object) -> Iterable[str]:
    if isinstance(value, list):
        for item in value:
            text = clean_text(item)
            if text:
                yield text
    else:
        text = clean_text(value)
        if text:
            yield text


def load_hc3() -> pd.DataFrame:
    subsets = ["finance", "medicine", "open_qa", "reddit_eli5", "wiki_csai"]
    records = []
    for subset in subsets:
        url = f"https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/{subset}.jsonl"
        print(f"Loading HC3 subset: {subset}")
        with urlopen(url, timeout=60) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line.decode("utf-8"))
                for text in flatten_answers(row.get("human_answers")):
                    records.append({"text": text, "label": 0, "source": f"Hello-SimpleAI/HC3:{subset}"})
                for text in flatten_answers(row.get("chatgpt_answers")):
                    records.append({"text": text, "label": 1, "source": f"Hello-SimpleAI/HC3:{subset}"})
    if not records:
        raise RuntimeError("Could not load any HC3 subset. Check network access.")
    frame = pd.DataFrame(records)
    print(f"Loaded {len(frame):,} labeled rows from Hello-SimpleAI/HC3.")
    return frame


def demo_frame() -> pd.DataFrame:
    ai_texts = [
        "It is important to note that artificial intelligence has transformed numerous industries through scalable automated decision making and comprehensive optimization strategies.",
        "Furthermore, the integration of machine learning systems enables organizations to unlock efficiencies, streamline workflows, and generate actionable insights across domains.",
        "In conclusion, responsible AI adoption requires robust governance, transparent evaluation, and a holistic understanding of technological and societal implications.",
    ]
    human_texts = [
        "I tried fixing my sink yesterday and made it worse. The first video looked easy, then water started leaking under the cabinet and I had to call my uncle.",
        "My teacher asked us to write about a book we actually liked, so I chose the one I read on the bus last month because the ending annoyed me for days.",
        "The cafe near my house changed its opening time again, which is small news, but it ruined my morning because I had counted on coffee before the train.",
    ]
    return pd.DataFrame(
        {
            "text": ai_texts + human_texts,
            "label": [1] * len(ai_texts) + [0] * len(human_texts),
            "source": ["demo"] * 6,
        }
    )


def prepare_dataset(frame: pd.DataFrame, min_words: int, max_rows: int | None) -> pd.DataFrame:
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(int)
    frame["word_count"] = frame["text"].str.split().map(len)
    frame = frame[(frame["word_count"] >= min_words) & frame["label"].isin([0, 1])]
    frame = frame.drop_duplicates(subset=["text"])

    if max_rows and len(frame) > max_rows:
        per_class = max_rows // 2
        sampled = []
        for label in [0, 1]:
            class_rows = frame[frame["label"] == label]
            sampled.append(class_rows.sample(min(per_class, len(class_rows)), random_state=42))
        frame = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42)

    counts = frame["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise RuntimeError("Training requires at least two usable human and two usable AI samples.")
    return frame


def train(args: argparse.Namespace) -> dict:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for slug in args.kaggle_dataset:
        download_kaggle_dataset(slug, data_dir)

    frames = [load_local_files(data_dir)]
    if args.include_hc3:
        frames.append(load_hc3())
    if args.allow_demo_data and all(frame.empty for frame in frames):
        print("No real dataset found; using tiny demo data because --allow-demo-data was set.")
        frames.append(demo_frame())

    dataset = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    dataset = prepare_dataset(dataset, min_words=args.min_words, max_rows=args.max_rows)

    print("\nDataset summary")
    print(dataset["label"].map({0: "human", 1: "ai"}).value_counts().to_string())
    print(dataset.groupby(["source", "label"]).size().head(20).to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=args.test_size,
        random_state=42,
        stratify=dataset["label"],
    )

    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=args.word_features,
                    min_df=args.min_df,
                    max_df=0.92,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=args.char_features,
                    min_df=args.min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=args.c,
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]
    report = classification_report(y_test, y_pred, target_names=["human", "ai"], output_dict=True)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_ai": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall_ai": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1_ai": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": report,
    }

    print("\nValidation metrics")
    for key in ["accuracy", "precision_ai", "recall_ai", "f1_ai", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4f}")
    print(classification_report(y_test, y_pred, target_names=["human", "ai"]))

    with open(args.model_out, "wb") as handle:
        pickle.dump(model, handle)
    with open(args.vectorizer_out, "wb") as handle:
        pickle.dump(vectorizer, handle)

    metadata = {
        "model_version": args.model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(dataset)),
        "test_rows": int(len(y_test)),
        "label_counts": {str(k): int(v) for k, v in dataset["label"].value_counts().items()},
        "sources": sorted(dataset["source"].unique().tolist()),
        "metrics": metrics,
        "threshold_policy": {
            "low_ai_signal": "score < 70 with enough reliability",
            "mixed_or_inconclusive": "40 <= score < 70 or low reliability",
            "elevated_ai_signal": "score >= 70 with enough reliability",
            "strong_statistical_evidence": "score >= 85 or <= 15 with high reliability; still not proof of authorship",
        },
        "responsible_use": (
            "Dataset-backed AI text detection can support review, but should not be treated as final proof. "
            "Use alongside drafts, citations, version history, and author discussion."
        ),
    }
    with open(args.metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\nSaved {args.model_out}, {args.vectorizer_out}, and {args.metadata_out}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OriginScript AI text detector from real labeled datasets.")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing Kaggle/GitHub CSV or JSONL files.")
    parser.add_argument("--kaggle-dataset", action="append", default=[], help="Kaggle slug, for example shanegerami/ai-vs-human-text.")
    parser.add_argument("--include-hc3", action="store_true", help="Load Hello-SimpleAI/HC3 human vs ChatGPT data.")
    parser.add_argument("--allow-demo-data", action="store_true", help="Use tiny demo data only when no real files are found.")
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--word-features", type=int, default=80000)
    parser.add_argument("--char-features", type=int, default=60000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--model-version", default="tfidf-word-char-logreg-dataset-v2")
    parser.add_argument("--model-out", default="model.pkl")
    parser.add_argument("--vectorizer-out", default="vectorizer.pkl")
    parser.add_argument("--metadata-out", default="model_metadata.json")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
