"""
AI Writing Review - Flask Backend

Portfolio-friendly MVP for uncertainty-aware AI-writing risk assessment.
The service reports statistical signals for human review; it does not claim
authorship or academic misconduct.
"""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import secrets
import sqlite3
import os
import pickle
import re
from werkzeug.security import check_password_hash, generate_password_hash

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
try:
    import stripe
except ImportError:
    stripe = None

from services.writing_assistant.citation_guard import validate_citations
from services.writing_assistant.factual_consistency import check_factual_consistency
from services.writing_assistant.passage_analyzer import analyze_passage
from services.writing_assistant.readability_checker import check_readability
from services.writing_assistant.revision_history import find_revision, get_history, save_revision, update_revision
from services.writing_assistant.revision_planner import build_revision_plan
from services.writing_assistant.rewrite_generator import generate_revision
from services.writing_assistant.semantic_validator import validate_similarity


app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

print("Loading model...")
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model files not found. Run 'python train_model.py' first.")
    raise SystemExit(1)

try:
    with open("model_metadata.json", "r", encoding="utf-8") as f:
        MODEL_METADATA = json.load(f)
except FileNotFoundError:
    MODEL_METADATA = {
        "model_version": "tfidf-logreg-demo-v1",
        "training_rows": None,
        "sources": ["legacy local model"],
        "metrics": {},
        "responsible_use": (
            "This detector reports statistical evidence only. It should not be treated as final proof "
            "of authorship or misconduct."
        ),
    }


SUPPORTED_FORMATS = ["PDF", "DOC", "DOCX", "TXT", "RTF", "ODT", "HTML", "EPUB"]
MIN_ELIGIBLE_WORDS = 300
ANALYSIS_CACHE = {}
STRIPE_API_VERSION = "2026-02-25.clover"
DB_PATH = os.getenv("APP_DB_PATH", "app_data.db")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                plan TEXT NOT NULL DEFAULT 'free',
                created_at TEXT NOT NULL,
                last_active_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                event_type TEXT NOT NULL,
                document_id TEXT,
                analysis_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_password = os.getenv("ADMIN_PASSWORD")
        existing = (
            conn.execute("SELECT id FROM users WHERE email = ?", (admin_email,)).fetchone()
            if admin_email
            else None
        )
        if admin_email and admin_password and existing is None:
            conn.execute(
                """
                INSERT INTO users (name, email, password_hash, role, plan, created_at)
                VALUES (?, ?, ?, 'admin', 'premium', ?)
                """,
                (
                    "Site Admin",
                    admin_email,
                    generate_password_hash(admin_password),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )


def row_to_user(row):
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "role": row["role"],
        "plan": row["plan"],
        "created_at": row["created_at"],
        "last_active_at": row["last_active_at"],
    }


def get_auth_user():
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header.removeprefix("Bearer ").strip()
    with db() as conn:
        row = conn.execute(
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    return row_to_user(row)


def log_usage(event_type, user_id=None, document_id=None, analysis_id=None, metadata=None):
    if user_id is None:
        user = get_auth_user()
        user_id = user["id"] if user else None
    with db() as conn:
        conn.execute(
            """
            INSERT INTO usage_records (user_id, event_type, document_id, analysis_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                event_type,
                document_id,
                analysis_id,
                metadata,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        if user_id:
            conn.execute(
                "UPDATE users SET last_active_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), user_id),
            )


init_db()


def clean_text(text):
    """Clean text for the TF-IDF model."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def convert_to_native(obj):
    """Recursively convert numpy values to JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_words(text):
    return re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)


def analyze_text_style(text):
    """Stylometric and likelihood-inspired heuristic features."""
    sentences = split_sentences(text)
    words = re.findall(r"\b[a-z']+\b", text.lower())

    if not words or not sentences:
        return {
            "perplexity": 0.0,
            "perplexity_score": 50.0,
            "burstiness": 0.0,
            "vocab_diversity": 0.0,
            "repetition": 0.0,
        }

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    entropy = 0.0
    for count in freq.values():
        p = count / len(words)
        if p > 0:
            entropy -= p * np.log2(p)

    max_entropy = np.log2(len(freq)) if len(freq) > 1 else 1
    perplexity_value = np.pow(2, entropy)
    perplexity_score = (1 - entropy / max_entropy) * 100 if max_entropy > 0 else 50

    sentence_lengths = [len(get_words(sentence)) for sentence in sentences]
    mean_len = np.mean(sentence_lengths)
    std_len = np.std(sentence_lengths)
    burstiness = (std_len / mean_len * 100) if mean_len > 0 else 0

    unique_words = len(set(words))
    vocab_diversity = unique_words / len(words) * 100

    bigrams = {}
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        bigrams[bigram] = bigrams.get(bigram, 0) + 1
    repeated = sum(count - 1 for count in bigrams.values() if count > 1)
    repetition = repeated / len(bigrams) * 100 if bigrams else 0

    return {
        "perplexity": float(perplexity_value),
        "perplexity_score": float(perplexity_score),
        "burstiness": float(min(100, burstiness)),
        "vocab_diversity": float(vocab_diversity),
        "repetition": float(min(100, repetition * 2)),
    }


def classify_sections(text):
    """Estimate how much input is eligible long-form prose."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = get_words(text)
    if not lines:
        return {
            "eligible_words": 0,
            "total_words": len(words),
            "bullet_ratio": 0.0,
            "table_like_ratio": 0.0,
            "quote_ratio": 0.0,
        }

    bullet_lines = sum(
        1 for line in lines if re.match(r"^(\*|-|--|\d+\.|\d+\)|[a-zA-Z]\))\s+", line)
    )
    table_like_lines = sum(1 for line in lines if line.count("|") >= 2 or line.count("\t") >= 2)
    quote_lines = sum(1 for line in lines if line.startswith((">", '"', "'")))

    ineligible_words = 0
    for line in lines:
        line_words = len(get_words(line))
        code_like = len(re.findall(r"[{}();=<>]", line)) > max(4, line_words // 2)
        list_like = re.match(r"^(\*|-|--|\d+\.|\d+\)|[a-zA-Z]\))\s+", line)
        table_like = line.count("|") >= 2 or line.count("\t") >= 2
        if list_like or table_like or code_like:
            ineligible_words += line_words

    return {
        "eligible_words": max(0, len(words) - ineligible_words),
        "total_words": len(words),
        "bullet_ratio": bullet_lines / len(lines),
        "table_like_ratio": table_like_lines / len(lines),
        "quote_ratio": quote_lines / len(lines),
    }


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def segment_passages(text, target_max=260, min_passage=80):
    """Split prose into non-overlapping sentence-bound passages for UI highlights."""
    sentences = split_sentences(text)
    passages = []
    current = []
    current_words = 0
    cursor = 0

    for sentence in sentences:
        count = len(get_words(sentence))
        if current and current_words + count > target_max and current_words >= min_passage:
            passage_text = " ".join(current).strip()
            start = text.find(current[0], cursor)
            end = start + len(passage_text) if start >= 0 else cursor + len(passage_text)
            passages.append((passage_text, max(0, start), max(0, end), current_words))
            cursor = max(cursor, end)

            current = []
            current_words = 0

        current.append(sentence)
        current_words += count

    if current:
        passage_text = " ".join(current).strip()
        if current_words >= min_passage or not passages:
            start = text.find(current[0], cursor)
            end = start + len(passage_text) if start >= 0 else cursor + len(passage_text)
            passages.append((passage_text, max(0, start), max(0, end), current_words))

    return passages


def score_text(text):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(tfidf)[0]
    ml_score = float(probabilities[1] * 100)
    model_confidence = float(abs(probabilities[1] - 0.5) * 200)
    style = analyze_text_style(text)
    heuristic_score = (
        style["perplexity_score"] * 0.3
        + (100 - style["burstiness"]) * 0.2
        + (100 - style["vocab_diversity"]) * 0.2
        + style["repetition"] * 0.3
    )
    final_score = float(ml_score * 0.7 + heuristic_score * 0.3)
    disagreement = abs(ml_score - heuristic_score)
    reliability = max(20.0, min(95.0, model_confidence - disagreement * 0.25 + 25.0))
    return final_score, ml_score, heuristic_score, model_confidence, reliability, style, probabilities


def classification_for(score, reliability):
    if reliability < 45 or 40 <= score < 70:
        return "mixed_or_inconclusive", "Mixed or inconclusive evidence"
    if score >= 70:
        return "elevated_ai_signal", "Elevated AI-writing signal"
    return "low_ai_signal", "Low AI-writing signal"


def confidence_label(reliability):
    if reliability >= 70:
        return "high"
    if reliability >= 45:
        return "moderate"
    return "low"


def passage_level(score, reliability):
    if reliability < 45 or 45 <= score < 65:
        return "uncertain"
    if score >= 65:
        return "elevated"
    return "low"


def evidence_strength(score, reliability):
    if reliability < 45:
        return "limited_statistical_evidence", "Limited statistical evidence"
    if score >= 85 and reliability >= 70:
        return "strong_ai_statistical_evidence", "Strong statistical evidence of AI-like writing"
    if score <= 15 and reliability >= 70:
        return "strong_human_statistical_evidence", "Strong statistical evidence of human-like writing"
    if score >= 70:
        return "moderate_ai_statistical_evidence", "Moderate statistical evidence of AI-like writing"
    if score <= 30:
        return "moderate_human_statistical_evidence", "Moderate statistical evidence of human-like writing"
    return "inconclusive_statistical_evidence", "Inconclusive statistical evidence"


def explain_passage(text, score, reliability, style):
    lower = text.lower()
    reasons = []
    ai_phrases = [
        "it is important to note",
        "furthermore",
        "moreover",
        "in conclusion",
        "in summary",
        "to summarize",
        "delve into",
        "plays a crucial role",
        "it is essential to",
        "in the context of",
        "it is evident that",
        "paramount",
        "multifaceted",
        "comprehensive",
    ]
    if any(phrase in lower for phrase in ai_phrases):
        reasons.append("Contains formulaic transition or framing phrases associated with generated prose.")
    if style["burstiness"] < 25:
        reasons.append("Sentence lengths are unusually even, reducing natural burstiness.")
    if style["vocab_diversity"] < 45:
        reasons.append("Vocabulary diversity is low for the passage length.")
    if style["repetition"] > 8:
        reasons.append("Repeated word-pair patterns increase the statistical signal.")
    if not reasons:
        reasons.append("The models found a statistical pattern match, but no single feature is conclusive.")

    limitations = []
    if reliability < 50:
        limitations.append("Model agreement is limited for this passage.")
    if len(get_words(text)) < 120:
        limitations.append("Short passages carry higher uncertainty.")
    if score > 60 and reliability < 60:
        limitations.append("Elevated score should be interpreted as review priority, not proof.")

    return reasons[:3], limitations


def get_top_features(text, top_n=10):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    tfidf_array = tfidf.toarray()[0]
    contributions = []

    for idx, value in enumerate(tfidf_array):
        if value > 0:
            contribution = float(coefficients[idx] * value)
            contributions.append(
                {
                    "term": str(feature_names[idx]),
                    "contribution": contribution,
                    "is_ai": bool(contribution > 0),
                }
            )

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_n]


def analyze_sentences(text):
    results = []
    for sentence in split_sentences(text):
        if len(clean_text(sentence)) < 5:
            score = 50.0
        else:
            try:
                score, _, _, _, _, _, _ = score_text(sentence)
            except Exception:
                score = 50.0

        level = "high-ai" if score >= 65 else "medium-ai" if score >= 45 else "low-ai"
        results.append({"text": str(sentence), "score": float(round(score, 1)), "level": level})
    return results


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or len(password) < 6:
        return jsonify({"error": "Name, email, and a password of at least 6 characters are required."}), 400
    try:
        with db() as conn:
            conn.execute(
                """
                INSERT INTO users (name, email, password_hash, role, plan, created_at, last_active_at)
                VALUES (?, ?, ?, 'user', 'free', ?, ?)
                """,
                (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        log_usage("account_created", user_id=row["id"])
        return jsonify({"success": True, "user": row_to_user(row)})
    except sqlite3.IntegrityError:
        return jsonify({"error": "An account with this email already exists."}), 409


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row is None or not check_password_hash(row["password_hash"], password):
            return jsonify({"error": "Invalid email or password."}), 401
        token = secrets.token_urlsafe(32)
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, row["id"], datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "UPDATE users SET last_active_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), row["id"]),
        )
    log_usage("login", user_id=row["id"])
    return jsonify({"success": True, "token": token, "user": row_to_user(row)})


@app.route("/auth/me")
def auth_me():
    user = get_auth_user()
    if not user:
        return jsonify({"error": "Not authenticated."}), 401
    return jsonify({"success": True, "user": user})


@app.route("/admin/usage")
def admin_usage():
    user = get_auth_user()
    if not user or user["role"] != "admin":
        return jsonify({"error": "Admin access required."}), 403
    with db() as conn:
        rows = conn.execute(
            """
            SELECT
                users.id,
                users.name,
                users.email,
                users.role,
                users.plan,
                users.last_active_at,
                COUNT(CASE WHEN usage_records.event_type = 'scan_completed' THEN 1 END) AS scans,
                COUNT(CASE WHEN usage_records.event_type = 'report_downloaded' THEN 1 END) AS reports,
                COUNT(CASE WHEN usage_records.event_type LIKE 'payment_%' THEN 1 END) AS payment_events
            FROM users
            LEFT JOIN usage_records ON usage_records.user_id = users.id
            GROUP BY users.id
            ORDER BY users.created_at DESC
            """
        ).fetchall()
        totals = conn.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM usage_records
            GROUP BY event_type
            ORDER BY count DESC
            """
        ).fetchall()
        pageviews = conn.execute(
            "SELECT metadata, created_at FROM usage_records WHERE event_type = 'page_view' ORDER BY created_at DESC"
        ).fetchall()
    visitor_ids = set()
    last_pageview = None
    now = datetime.now(timezone.utc)
    bucket_count = 24
    bucket_minutes = 5
    buckets = []
    for index in range(bucket_count):
        start = now - timedelta(minutes=bucket_minutes * (bucket_count - index))
        end = start + timedelta(minutes=bucket_minutes)
        buckets.append(
            {
                "label": start.strftime("%H:%M"),
                "start": start,
                "end": end,
                "page_views": 0,
                "visitor_ids": set(),
            }
        )

    for row in pageviews:
        last_pageview = last_pageview or row["created_at"]
        metadata = row["metadata"] or ""
        row_visitor_id = ""
        for part in metadata.split(";"):
            if part.startswith("visitor_id=") and part.removeprefix("visitor_id="):
                row_visitor_id = part.removeprefix("visitor_id=")
                visitor_ids.add(row_visitor_id)
        try:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        except ValueError:
            created_at = None
        if created_at:
            for bucket in buckets:
                if bucket["start"] <= created_at < bucket["end"]:
                    bucket["page_views"] += 1
                    if row_visitor_id:
                        bucket["visitor_ids"].add(row_visitor_id)
                    break

    traffic_timeline = [
        {
            "label": bucket["label"],
            "page_views": bucket["page_views"],
            "unique_visitors": len(bucket["visitor_ids"]),
        }
        for bucket in buckets
    ]
    return jsonify(
        {
            "success": True,
            "users": [dict(row) for row in rows],
            "totals": [dict(row) for row in totals],
            "traffic": {
                "page_views": len(pageviews),
                "unique_visitors": len(visitor_ids),
                "last_pageview": last_pageview,
                "timeline": traffic_timeline,
            },
        }
    )


@app.route("/analyses/<analysis_id>/report-download", methods=["POST"])
def report_downloaded(analysis_id):
    analysis = ANALYSIS_CACHE.get(analysis_id)
    log_usage(
        "report_downloaded",
        document_id=analysis["document_id"] if analysis else None,
        analysis_id=analysis_id,
    )
    return jsonify({"success": True})


@app.route("/traffic/pageview", methods=["POST"])
def traffic_pageview():
    data = request.get_json() or {}
    log_usage(
        "page_view",
        metadata=(
            f"path={data.get('path', '/')};"
            f"visitor_id={data.get('visitor_id', '')};"
            f"referrer={data.get('referrer', '')}"
        ),
    )
    return jsonify({"success": True})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        filename = data.get("filename") or "pasted-text.txt"
        mime_type = data.get("mime_type") or "text/plain"
        retention = data.get("retention") or "standard_30_days"

        if len(text) < 50:
            return jsonify({"error": "Text too short. Please provide at least 50 characters."}), 400

        cleaned = clean_text(text)
        if len(cleaned) < 20:
            return jsonify({"error": "Text contains too little meaningful content."}), 400

        sections = classify_sections(text)
        limitations = []
        eligibility_status = "eligible"
        if sections["eligible_words"] < MIN_ELIGIBLE_WORDS:
            eligibility_status = "insufficient_text"
            limitations.append(
                f"Only {sections['eligible_words']} eligible words found; {MIN_ELIGIBLE_WORDS} are recommended."
            )
        if sections["bullet_ratio"] > 0.45:
            limitations.append("A high proportion of the document appears to be bullet points.")
        if sections["table_like_ratio"] > 0.25:
            limitations.append("A high proportion of the document appears table-like.")
        if sections["quote_ratio"] > 0.25:
            limitations.append("A high proportion of the document may be quoted material.")

        (
            final_score,
            ml_score,
            heuristic_score,
            model_confidence,
            reliability,
            style,
            probabilities,
        ) = score_text(text)

        if abs(ml_score - heuristic_score) > 25:
            limitations.append("Dataset model and stylometric signals disagree.")
        if style["vocab_diversity"] < 35:
            limitations.append("Low vocabulary diversity can reduce confidence.")

        classification, classification_label = classification_for(final_score, reliability)
        evidence_code, evidence_label = evidence_strength(final_score, reliability)
        confidence = confidence_label(reliability)

        passage_results = []
        for index, (passage_text, start, end, word_count) in enumerate(segment_passages(text), start=1):
            (
                p_score,
                p_ml,
                p_heuristic,
                _,
                p_reliability,
                p_style,
                _,
            ) = score_text(passage_text)
            reasons, passage_limitations = explain_passage(passage_text, p_score, p_reliability, p_style)
            passage_results.append(
                {
                    "passage_id": f"pas_{index:03d}",
                    "page_start": None,
                    "page_end": None,
                    "character_start": start,
                    "character_end": end,
                    "word_count": word_count,
                    "text": passage_text,
                    "ai_signal": round(p_score, 1),
                    "confidence": confidence_label(p_reliability),
                    "reliability": round(p_reliability, 1),
                    "level": passage_level(p_score, p_reliability),
                    "model_agreement": round(100 - abs(p_ml - p_heuristic), 1),
                    "explanations": reasons,
                    "limitations": passage_limitations,
                }
            )

        now = datetime.now(timezone.utc)
        doc_hash = sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        response_data = {
            "success": True,
            "analysis_id": f"ana_{doc_hash[:10]}",
            "document_id": f"doc_{doc_hash[:10]}",
            "generated_at": now.isoformat(),
            "model_version": MODEL_METADATA.get("model_version", "tfidf-logreg-demo-v1"),
            "model_training": {
                "training_rows": MODEL_METADATA.get("training_rows"),
                "test_rows": MODEL_METADATA.get("test_rows"),
                "sources": MODEL_METADATA.get("sources", []),
                "metrics": MODEL_METADATA.get("metrics", {}),
            },
            "eligibility_status": eligibility_status,
            "ai_score": int(round(final_score)),
            "ai_signal": round(final_score / 100, 3),
            "human_likelihood": int(round(100 - final_score)),
            "ml_score": round(ml_score, 1),
            "heuristic_score": round(heuristic_score, 1),
            "confidence": confidence,
            "confidence_score": round(reliability, 1),
            "model_confidence": round(model_confidence, 1),
            "classification": classification,
            "classification_label": classification_label,
            "evidence_strength": evidence_code,
            "evidence_strength_label": evidence_label,
            "verdict": classification_label,
            "probabilities": {
                "human": round(probabilities[0] * 100, 1),
                "ai": round(probabilities[1] * 100, 1),
            },
            "style_metrics": {
                "perplexity": round(style["perplexity"], 1),
                "burstiness": round(style["burstiness"], 1),
                "vocab_diversity": round(style["vocab_diversity"], 1),
                "repetition": round(style["repetition"], 1),
            },
            "document": {
                "filename": filename,
                "mime_type": mime_type,
                "sha256": doc_hash,
                "retention": retention,
                "word_count": sections["total_words"],
                "eligible_word_count": sections["eligible_words"],
                "supported_formats": SUPPORTED_FORMATS,
                "minimum_eligible_words": MIN_ELIGIBLE_WORDS,
            },
            "limitations": limitations
            or ["No major input-quality limitations were detected by this demo pipeline."],
            "top_features": get_top_features(text, 12),
            "sentence_analysis": analyze_sentences(text),
            "passages": passage_results,
            "processing": [
                {"label": "Validate file", "status": "completed"},
                {"label": "Extract text", "status": "completed"},
                {"label": "Identify eligible prose", "status": "completed"},
                {"label": "Segment passages", "status": "completed"},
                {"label": "Run detection models", "status": "completed"},
                {"label": "Calibrate score", "status": "completed"},
                {"label": "Generate report", "status": "completed"},
            ],
            "disclaimer": (
                "This report identifies dataset-backed statistical patterns associated with AI-generated writing. "
                "Even strong statistical evidence is not final proof of authorship or academic misconduct. "
                "Review it alongside drafts, citations, writing history, and discussion with the author."
            ),
            "word_count": sections["total_words"],
            "char_count": len(text),
        }

        ANALYSIS_CACHE[response_data["analysis_id"]] = response_data
        log_usage(
            "scan_completed",
            document_id=response_data["document_id"],
            analysis_id=response_data["analysis_id"],
            metadata=f"filename={filename};score={response_data['ai_score']};eligible_words={sections['eligible_words']}",
        )

        return jsonify(convert_to_native(response_data))
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "features": int(len(vectorizer.get_feature_names_out())),
            "model_version": MODEL_METADATA.get("model_version", "tfidf-logreg-demo-v1"),
            "training_rows": MODEL_METADATA.get("training_rows"),
            "validation_metrics": MODEL_METADATA.get("metrics", {}),
            "supported_formats": SUPPORTED_FORMATS,
            "minimum_eligible_words": MIN_ELIGIBLE_WORDS,
        }
    )


@app.route("/payments/create-checkout-session", methods=["POST"])
def create_checkout_session():
    if stripe is None:
        return jsonify({"error": "Stripe SDK is not installed. Run pip install -r requirements.txt."}), 500

    secret_key = os.getenv("STRIPE_SECRET_KEY")
    if not secret_key:
        return jsonify({"error": "STRIPE_SECRET_KEY is not configured on the server."}), 500

    stripe.api_key = secret_key
    stripe.api_version = STRIPE_API_VERSION

    data = request.get_json() or {}
    origin = data.get("origin") or request.headers.get("Origin") or request.host_url.rstrip("/")
    customer_email = data.get("email") or None
    price_id = os.getenv("STRIPE_PREMIUM_PRICE_ID")

    line_item = {"quantity": 1}
    if price_id:
        line_item["price"] = price_id
    else:
        line_item["price_data"] = {
            "currency": "eur",
            "unit_amount": 500,
            "recurring": {"interval": "month"},
            "product_data": {
                "name": "OriginScript AI Premium",
                "description": "Writing Improvement Assistant, comparison, reanalysis, and export tools.",
            },
        }

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=customer_email,
            line_items=[line_item],
            success_url=f"{origin}?payment=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{origin}?payment=cancelled",
            metadata={"plan": "premium", "product": "ai-detection-system"},
        )
        log_usage("payment_checkout_started", metadata=f"session_id={session.id};email={customer_email or ''}")
        return jsonify({"success": True, "checkout_url": session.url, "session_id": session.id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def build_rewrite_response(analysis_id, approve=False):
    data = request.get_json() or {}
    analysis = ANALYSIS_CACHE.get(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found in this demo session. Re-run analysis first."}), 404

    passage_ids = data.get("passage_ids") or []
    options = {
        "mode": data.get("mode", "guided"),
        "tone": data.get("tone", "academic"),
        "complexity": data.get("complexity", "balanced"),
        "revision_strength": data.get("revision_strength", "moderate"),
        "preserve_citations": bool(data.get("preserve_citations", True)),
        "preserve_technical_terms": bool(data.get("preserve_technical_terms", True)),
        "shorter_sentences": bool(data.get("shorter_sentences", False)),
        "add_examples": bool(data.get("add_examples", False)),
        "add_evidence_prompts": bool(data.get("add_evidence_prompts", True)),
    }
    user_context = data.get("user_context") or {}

    selected = [
        passage for passage in analysis["passages"]
        if not passage_ids or passage["passage_id"] in passage_ids
    ]
    if not selected:
        return jsonify({"error": "No matching passages selected."}), 400

    candidates = []
    for passage in selected:
        plan = build_revision_plan(passage["text"], options)
        generated = generate_revision(passage["text"], options, user_context)
        revised = generated["revised_text"]
        citation_check = validate_citations(passage["text"], revised)
        similarity_check = validate_similarity(passage["text"], revised)
        factual_check = check_factual_consistency(passage["text"], revised)
        readability = check_readability(revised)
        candidates.append(
            {
                "passage_id": passage["passage_id"],
                "original_text": passage["text"],
                "revised_text": revised,
                "analysis": analyze_passage(passage["text"]),
                "revision_plan": plan,
                "change_summary": generated["change_summary"],
                "quality_checks": {
                    "meaning_preservation": similarity_check,
                    "citation_preservation": citation_check,
                    "factual_consistency": factual_check,
                    "readability": readability,
                    "user_review": {
                        "status": "required",
                        "note": "Author approval is required before exporting or reanalysis.",
                    },
                },
            }
        )

    record = save_revision(
        analysis["document_id"],
        analysis_id,
        {
            "mode": options["mode"],
            "tone": options["tone"],
            "options": options,
            "user_context": user_context,
            "candidates": candidates,
        },
    )
    if approve:
        update_revision(record["revision_id"], "approved")
        record["status"] = "approved"

    return jsonify(
        convert_to_native(
            {
                "success": True,
                "revision_id": record["revision_id"],
                "status": record["status"],
                "document_id": analysis["document_id"],
                "analysis_id": analysis_id,
                "candidates": candidates,
                "disclaimer": (
                    "Writing improvement is for clarity, evidence, and author review. "
                    "It is not designed to bypass AI detectors, and reanalysis may not lower the score."
                ),
            }
        )
    )


@app.route("/analyses/<analysis_id>/rewrite-preview", methods=["POST"])
def rewrite_preview(analysis_id):
    return build_rewrite_response(analysis_id, approve=False)


@app.route("/analyses/<analysis_id>/rewrite", methods=["POST"])
def rewrite(analysis_id):
    return build_rewrite_response(analysis_id, approve=False)


@app.route("/analyses/<analysis_id>/rewrite-guided", methods=["POST"])
def rewrite_guided(analysis_id):
    return build_rewrite_response(analysis_id, approve=False)


@app.route("/revisions/<revision_id>/approve", methods=["POST"])
def approve_revision(revision_id):
    record = update_revision(revision_id, "approved")
    if not record:
        return jsonify({"error": "Revision not found."}), 404
    return jsonify(convert_to_native({"success": True, "revision": record}))


@app.route("/revisions/<revision_id>/reject", methods=["POST"])
def reject_revision(revision_id):
    record = update_revision(revision_id, "rejected")
    if not record:
        return jsonify({"error": "Revision not found."}), 404
    return jsonify(convert_to_native({"success": True, "revision": record}))


@app.route("/revisions/<revision_id>/reanalyze", methods=["POST"])
def reanalyze_revision(revision_id):
    data = request.get_json() or {}
    if not find_revision(revision_id):
        return jsonify({"error": "Revision not found."}), 404
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Provide revised text for reanalysis."}), 400
    with app.test_request_context("/analyze", method="POST", json={"text": text, "filename": "revised-document.txt"}):
        return analyze()


@app.route("/documents/<document_id>/revision-history")
def revision_history(document_id):
    return jsonify(convert_to_native({"success": True, "history": get_history(document_id)}))


@app.route("/revisions/<revision_id>/comparison")
def revision_comparison(revision_id):
    record = find_revision(revision_id)
    if record:
        return jsonify(convert_to_native({"success": True, "revision": record}))
    return jsonify({"error": "Revision not found."}), 404


@app.route("/documents/<document_id>/export-revised", methods=["POST"])
def export_revised(document_id):
    data = request.get_json() or {}
    text = data.get("text", "")
    export_format = data.get("format", "txt")
    log_usage("revised_exported", document_id=document_id, metadata=f"format={export_format}")
    return jsonify(
        {
            "success": True,
            "document_id": document_id,
            "format": export_format,
            "content": text,
            "note": "This demo returns export content inline. Production can generate DOCX, PDF, TXT, and HTML files.",
        }
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AI WRITING REVIEW - FLASK SERVER")
    print("=" * 60)
    print("Model: TF-IDF + Logistic Regression demo ensemble")
    print(f"Features: {len(vectorizer.get_feature_names_out())}")
    print("Server starting on http://localhost:5000")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host="0.0.0.0", port=5000)
