import sqlite3
import json
from typing import List, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt

NODE_COLORS = {
    "Concept": "#4C78A8",
    "Topic": "#72B7B2",
    "Entity": "#F58518",
    "Flashcard": "#E45756",
}


def visualize_graph(db_path: str, output_path: str = "graph.png", max_nodes: int = 50) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, node_type, label, properties FROM nodes LIMIT ?", (max_nodes,))
    nodes = cur.fetchall()
    node_ids = {row[0] for row in nodes}

    cur.execute(
        "SELECT source_id, target_id, relationship_type FROM edges WHERE source_id IN ({ids}) AND target_id IN ({ids})".format(
            ids=",".join(str(i) for i in node_ids) if node_ids else "0"
        )
    )
    edges = cur.fetchall()

    G = nx.DiGraph()
    for n in nodes:
        nid, ntype, label = n[0], n[1], n[2]
        G.add_node(nid, label=label, type=ntype)

    for e in edges:
        src, tgt, rtype = e[0], e[1], e[2]
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt, type=rtype)

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    colors = [NODE_COLORS.get(G.nodes[n]["type"], "#B0BEC5") for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels, node_size=600, font_size=8, arrows=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_graph_json(db_path: str, output_path: str = "graph.json") -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, node_type, label, properties FROM nodes")
    nodes = [dict(row) for row in cur.fetchall()]
    # Convert properties from TEXT JSON to dict
    for n in nodes:
        try:
            n["properties"] = json.loads(n.get("properties") or "{}")
        except Exception:
            n["properties"] = {}

    cur.execute("SELECT source_id as source, target_id as target, relationship_type as type, weight FROM edges")
    edges = [dict(row) for row in cur.fetchall()]

    payload = {"nodes": nodes, "edges": edges}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
