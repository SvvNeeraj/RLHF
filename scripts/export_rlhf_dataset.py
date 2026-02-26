import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def load_from_feedback_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not obj.get("query") or not obj.get("response"):
            continue
        rows.append(
            {
                "user_id": obj.get("user_id", ""),
                "query": obj.get("query", ""),
                "query_key": obj.get("query_key", ""),
                "response": obj.get("response", ""),
                "rating": int(obj.get("rating", 0)),
                "reward": float(obj.get("reward", 0.0)),
                "request_style": obj.get("request_style", ""),
                "timestamp": obj.get("timestamp", ""),
            }
        )
    return rows


def load_from_sqlite(sqlite_path: Path) -> list[dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            f.user_id,
            c.query,
            c.query_key,
            c.response,
            f.rating,
            f.reward,
            c.request_style,
            f.created_at AS timestamp
        FROM feedback_log f
        JOIN chat_history c ON c.id = f.chat_id
        ORDER BY f.id ASC
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_preference_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        qk = str(row.get("query_key", "")).strip().lower()
        if not qk:
            continue
        by_key.setdefault(qk, []).append(row)

    pairs: list[dict[str, Any]] = []
    for qk, items in by_key.items():
        pos = sorted([x for x in items if float(x.get("reward", 0.0)) >= 0.3], key=lambda z: float(z.get("reward", 0.0)), reverse=True)
        neg = sorted([x for x in items if float(x.get("reward", 0.0)) <= -0.3], key=lambda z: float(z.get("reward", 0.0)))
        if not pos or not neg:
            continue
        p = pos[0]
        n = neg[0]
        pairs.append(
            {
                "query_key": qk,
                "query": p.get("query") or n.get("query"),
                "chosen": p.get("response", ""),
                "rejected": n.get("response", ""),
                "chosen_reward": float(p.get("reward", 0.0)),
                "rejected_reward": float(n.get("reward", 0.0)),
                "chosen_rating": int(p.get("rating", 0)),
                "rejected_rating": int(n.get("rating", 0)),
                "request_style": p.get("request_style") or n.get("request_style", ""),
            }
        )
    return pairs


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        query = str(r.get("query", "")).strip()
        response = str(r.get("response", "")).strip()
        if not query or not response:
            continue
        out.append(
            {
                "user_id": str(r.get("user_id", "")).strip(),
                "query": query,
                "query_key": str(r.get("query_key", "")).strip().lower(),
                "response": response,
                "rating": int(r.get("rating", 0)),
                "reward": float(r.get("reward", 0.0)),
                "request_style": str(r.get("request_style", "")).strip(),
                "timestamp": str(r.get("timestamp", "")).strip(),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RLHF datasets from feedback logs.")
    parser.add_argument("--sqlite", default="academic_ai.db", help="Path to SQLite database.")
    parser.add_argument("--feedback-jsonl", default="data/rlhf/feedback_events.jsonl", help="Path to feedback events jsonl.")
    parser.add_argument("--out-train", default="data/rlhf/policy_gradient_train.jsonl", help="Output JSONL for policy-gradient training.")
    parser.add_argument("--out-pairs", default="data/rlhf/preference_pairs.jsonl", help="Output JSONL preference pairs.")
    args = parser.parse_args()

    sqlite_rows = load_from_sqlite(Path(args.sqlite))
    jsonl_rows = load_from_feedback_jsonl(Path(args.feedback_jsonl))
    rows = normalize_rows(sqlite_rows or jsonl_rows)

    if not rows:
        print("No RLHF rows found. Collect feedback first.")
        return

    write_jsonl(Path(args.out_train), rows)
    pairs = build_preference_pairs(rows)
    write_jsonl(Path(args.out_pairs), pairs)

    print(f"Exported training rows: {len(rows)} -> {args.out_train}")
    print(f"Exported preference pairs: {len(pairs)} -> {args.out_pairs}")


if __name__ == "__main__":
    main()

