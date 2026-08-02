from datetime import datetime, timezone
from uuid import uuid4


REVISION_HISTORY = {}


def save_revision(document_id, analysis_id, payload):
    revision_id = f"rev_{uuid4().hex[:12]}"
    record = {
        "revision_id": revision_id,
        "document_id": document_id,
        "analysis_id": analysis_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "approved_at": None,
        "status": "candidate",
        **payload,
    }
    REVISION_HISTORY.setdefault(document_id, []).insert(0, record)
    return record


def update_revision(revision_id, status):
    for records in REVISION_HISTORY.values():
        for record in records:
            if record["revision_id"] == revision_id:
                record["status"] = status
                if status == "approved":
                    record["approved_at"] = datetime.now(timezone.utc).isoformat()
                return record
    return None


def get_history(document_id):
    return REVISION_HISTORY.get(document_id, [])


def find_revision(revision_id):
    for records in REVISION_HISTORY.values():
        for record in records:
            if record["revision_id"] == revision_id:
                return record
    return None
