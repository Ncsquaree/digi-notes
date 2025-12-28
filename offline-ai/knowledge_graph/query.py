import sqlite3
from typing import List, Dict, Any, Optional


def _get_node_by_label(conn: sqlite3.Connection, label: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM nodes WHERE label=?", (label,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def get_related_concepts(db_path: str, concept_label: str, max_depth: int = 2) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    start_id = _get_node_by_label(conn, concept_label)
    if start_id is None:
        return []

    cur = conn.cursor()
    visited = set([start_id])
    frontier = [(start_id, 0)]
    results: List[Dict[str, Any]] = []

    while frontier:
        nid, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        cur.execute("SELECT target_id, relationship_type, weight FROM edges WHERE source_id=?", (nid,))
        for row in cur.fetchall():
            tgt = int(row[0])
            if tgt in visited:
                continue
            visited.add(tgt)
            cur.execute("SELECT label FROM nodes WHERE id=?", (tgt,))
            lbl_row = cur.fetchone()
            label = lbl_row[0] if lbl_row else None
            results.append({"label": label, "relationship_type": row[1], "weight": float(row[2])})
            frontier.append((tgt, depth + 1))
    return results


def find_flashcards_for_concept(db_path: str, concept_label: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT f.id, f.question, f.answer, f.difficulty, f.context
        FROM flashcards f
        JOIN edges e ON e.source_id = (SELECT id FROM nodes WHERE label = 'flashcard-' || f.id)
        JOIN nodes n ON e.target_id = n.id
        WHERE n.label = ? AND e.relationship_type = 'DERIVED_FROM'
        """,
        (concept_label,),
    )
    return [dict(row) for row in cur.fetchall()]


def get_topic_hierarchy(db_path: str, topic_label: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    root_id = _get_node_by_label(conn, topic_label)
    if root_id is None:
        return {}

    def collect_children(node_id: int) -> Dict[str, Any]:
        # PART_OF edges are stored child → parent; traverse incoming edges to find children
        cur.execute("SELECT source_id FROM edges WHERE target_id=? AND relationship_type='PART_OF'", (node_id,))
        children = []
        for row in cur.fetchall():
            child_id = int(row[0])
            cur.execute("SELECT label FROM nodes WHERE id=?", (child_id,))
            lbl_row = cur.fetchone()
            child_label = lbl_row[0] if lbl_row else None
            children.append({"label": child_label, "children": collect_children(child_id)})
        return {"subtopics": children}

    return {"topic": topic_label, **collect_children(root_id)}
