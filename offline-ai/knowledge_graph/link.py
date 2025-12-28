"""
Semantic linking between flashcards and knowledge graph nodes.

Phase 7 Implementation:
  - link_flashcards_to_graph(): Connect flashcards to relevant KG nodes using embeddings
  - cluster_flashcards(): Group similar flashcards using K-means clustering
  - find_similar_flashcards_for_node(): Query API for flashcard retrieval
  - find_similar_nodes_for_flashcard(): Reverse query for node retrieval

Design:
  - Computes pairwise cosine similarity between flashcard and node embeddings
  - Creates edges with DERIVED_FROM/EXPLAINS relationship types
  - Threshold filtering (>0.7) with fallback linking for orphaned flashcards
  - Clustering uses sklearn K-means on embedding space
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
    logger.warning("scikit-learn not installed; clustering will be unavailable")


def _load_embeddings_for_entities(
    conn: sqlite3.Connection,
    entity_type: str,
    entity_ids: List[int]
) -> Dict[int, np.ndarray]:
    """Load embeddings from database for specified entities.
    
    Args:
        conn: SQLite database connection
        entity_type: Type of entity ('flashcard', 'node', etc.)
        entity_ids: List of entity IDs to load embeddings for
    
    Returns:
        Dict mapping entity_id to embedding vector (numpy array)
    """
    if not entity_ids:
        return {}
    
    embeddings_map = {}
    placeholders = ','.join('?' * len(entity_ids))
    query = f"""
        SELECT entity_id, vector 
        FROM embeddings 
        WHERE entity_type = ? AND entity_id IN ({placeholders})
    """
    
    try:
        cursor = conn.execute(query, (entity_type, *entity_ids))
        for entity_id, vec_blob in cursor.fetchall():
            if vec_blob:
                vec_array = np.frombuffer(vec_blob, dtype=np.float32)
                embeddings_map[entity_id] = vec_array
    except Exception as e:
        logger.error(f"Failed to load embeddings for {entity_type}: {e}")
    
    return embeddings_map


def _create_semantic_edges(
    conn: sqlite3.Connection,
    flashcard_id: int,
    node_id: int,
    similarity: float,
    relationship_type: str = 'DERIVED_FROM'
) -> bool:
    """Create semantic edge between flashcard and KG node if similarity threshold met.
    
    Args:
        conn: SQLite database connection
        flashcard_id: ID of the flashcard
        node_id: ID of the knowledge graph node
        similarity: Cosine similarity score
        relationship_type: Edge type ('DERIVED_FROM' or 'EXPLAINS')
    
    Returns:
        bool: True if edge was created, False otherwise
    """
    # Lookup flashcard node in nodes table (created by GraphBuilder)
    flashcard_node_query = conn.execute(
        "SELECT id FROM nodes WHERE label = ? LIMIT 1",
        (f"flashcard-{flashcard_id}",)
    ).fetchone()
    
    if not flashcard_node_query:
        logger.warning(f"Flashcard node not found for flashcard_id={flashcard_id}")
        return False
    
    flashcard_node_id = flashcard_node_query[0]
    
    # Check if edge already exists
    existing = conn.execute(
        """SELECT 1 FROM edges 
           WHERE source_id = ? AND target_id = ? AND relationship_type = ?""",
        (flashcard_node_id, node_id, relationship_type)
    ).fetchone()
    
    if existing:
        return False
    
    # Insert edge
    conn.execute(
        """INSERT INTO edges (source_id, target_id, relationship_type, weight)
           VALUES (?, ?, ?, ?)""",
        (flashcard_node_id, node_id, relationship_type, similarity)
    )
    
    return True


def _ensure_minimum_links(
    conn: sqlite3.Connection,
    flashcard_id: int,
    node_similarities: List[Tuple[int, float]],
    node_types: Dict[int, str]
) -> None:
    """Ensure flashcard has at least one link by creating edge to best match.
    
    Args:
        conn: SQLite database connection
        flashcard_id: ID of the flashcard
        node_similarities: List of (node_id, similarity) tuples sorted descending
        node_types: Mapping of node_id to node_type for relationship selection
    """
    # Check if flashcard already has links
    flashcard_node_query = conn.execute(
        "SELECT id FROM nodes WHERE label = ? LIMIT 1",
        (f"flashcard-{flashcard_id}",)
    ).fetchone()
    
    if not flashcard_node_query:
        return
    
    flashcard_node_id = flashcard_node_query[0]
    
    existing_links = conn.execute(
        """SELECT COUNT(*) FROM edges 
           WHERE source_id = ? AND relationship_type IN ('DERIVED_FROM', 'EXPLAINS')""",
        (flashcard_node_id,)
    ).fetchone()[0]
    
    if existing_links > 0:
        return
    
    # Create edge to best match
    if node_similarities:
        best_node_id, best_similarity = node_similarities[0]
        # Determine relationship type based on node type
        best_node_type = node_types.get(best_node_id)
        fallback_rel = 'EXPLAINS' if best_node_type == 'Topic' else 'DERIVED_FROM'
        
        conn.execute(
            """INSERT INTO edges (source_id, target_id, relationship_type, weight)
               VALUES (?, ?, ?, ?)""",
            (flashcard_node_id, best_node_id, fallback_rel, best_similarity)
        )
        logger.info(
            f"Fallback link created: flashcard {flashcard_id} → node {best_node_id} "
            f"(type={best_node_type}, similarity={best_similarity:.3f})"
        )


def link_flashcards_to_graph(
    db_path: str,
    similarity_threshold: float = 0.7,
    embedding_generator: Optional[Any] = None
) -> Dict[str, Any]:
    """Link flashcards to knowledge graph nodes using embedding-based semantic similarity.
    
    Creates edges between flashcards and KG nodes where cosine similarity exceeds threshold.
    Ensures every flashcard has at least one connection (fallback to best match).
    
    Args:
        db_path: Path to SQLite database
        similarity_threshold: Minimum cosine similarity to create edge (default: 0.7)
        embedding_generator: Optional EmbeddingGenerator instance for cosine_similarity
    
    Returns:
        Dict with keys:
            - links_created: Number of edges created
            - flashcards_processed: Number of flashcards processed
            - nodes_processed: Number of nodes processed
            - avg_similarity: Average similarity score of created edges
    
    Example:
        >>> from utils.embeddings import EmbeddingGenerator
        >>> emb_gen = EmbeddingGenerator(db_path="offline_ai.db")
        >>> result = link_flashcards_to_graph("offline_ai.db", similarity_threshold=0.7, embedding_generator=emb_gen)
        >>> print(f"Created {result['links_created']} semantic links")
    """
    conn = sqlite3.connect(db_path)
    
    try:
        # Import cosine_similarity function
        from utils.embeddings import cosine_similarity
        
        # Load flashcards with embeddings
        flashcard_query = conn.execute(
            """SELECT f.id, f.embedding_id 
               FROM flashcards f 
               WHERE f.embedding_id IS NOT NULL"""
        ).fetchall()
        
        flashcard_ids = [fc_id for fc_id, _ in flashcard_query]
        flashcard_embedding_ids = {fc_id: emb_id for fc_id, emb_id in flashcard_query}
        
        if not flashcard_ids:
            logger.warning("No flashcards with embeddings found")
            return {
                'links_created': 0,
                'flashcards_processed': 0,
                'nodes_processed': 0,
                'avg_similarity': 0.0
            }
        
        # Load KG nodes with embeddings (Concept, Topic, Entity types)
        node_query = conn.execute(
            """SELECT n.id, n.node_type, n.embedding_id 
               FROM nodes n 
               WHERE n.node_type IN ('Concept', 'Topic', 'Entity') 
               AND n.embedding_id IS NOT NULL"""
        ).fetchall()
        
        node_ids = [n_id for n_id, _, _ in node_query]
        node_types = {n_id: n_type for n_id, n_type, _ in node_query}
        node_embedding_ids = {n_id: emb_id for n_id, _, emb_id in node_query}
        
        if not node_ids:
            logger.warning("No KG nodes with embeddings found")
            return {
                'links_created': 0,
                'flashcards_processed': len(flashcard_ids),
                'nodes_processed': 0,
                'avg_similarity': 0.0
            }
        
        # Load embeddings from database
        flashcard_embeddings = {}
        for fc_id, emb_id in flashcard_embedding_ids.items():
            emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (emb_id,)).fetchone()
            if emb_row and emb_row[0]:
                flashcard_embeddings[fc_id] = np.frombuffer(emb_row[0], dtype=np.float32)
        
        node_embeddings = {}
        for n_id, emb_id in node_embedding_ids.items():
            emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (emb_id,)).fetchone()
            if emb_row and emb_row[0]:
                node_embeddings[n_id] = np.frombuffer(emb_row[0], dtype=np.float32)
        
        # Compute pairwise similarities and create edges
        edges_to_create = []
        similarity_scores = []
        flashcard_similarities = {}  # Track similarities for fallback
        
        for fc_id, fc_emb in flashcard_embeddings.items():
            node_sims = []
            
            for n_id, n_emb in node_embeddings.items():
                similarity = cosine_similarity(fc_emb.tolist(), n_emb.tolist())
                node_sims.append((n_id, similarity))
                
                if similarity > similarity_threshold:
                    # Determine relationship type based on node type
                    node_type = node_types[n_id]
                    rel_type = 'EXPLAINS' if node_type == 'Topic' else 'DERIVED_FROM'
                    
                    edges_to_create.append((fc_id, n_id, similarity, rel_type))
                    similarity_scores.append(similarity)
            
            # Sort by similarity for fallback
            node_sims.sort(key=lambda x: x[1], reverse=True)
            flashcard_similarities[fc_id] = node_sims
        
        # Batch insert edges
        links_created = 0
        for fc_id, n_id, similarity, rel_type in edges_to_create:
            if _create_semantic_edges(conn, fc_id, n_id, similarity, rel_type):
                links_created += 1
        
        # Ensure minimum links (fallback)
        for fc_id, node_sims in flashcard_similarities.items():
            _ensure_minimum_links(conn, fc_id, node_sims, node_types)
        
        conn.commit()
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        result = {
            'links_created': links_created,
            'flashcards_processed': len(flashcard_embeddings),
            'nodes_processed': len(node_embeddings),
            'avg_similarity': float(avg_similarity)
        }
        
        logger.info(f"Semantic linking complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to link flashcards to graph: {e}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()


def cluster_flashcards(
    db_path: str,
    n_clusters: int = 5,
    embedding_generator: Optional[Any] = None
) -> Dict[int, int]:
    """Cluster flashcards using K-means on embedding space.
    
    Groups similar flashcards together for spaced repetition optimization
    and topic-based organization.
    
    Args:
        db_path: Path to SQLite database
        n_clusters: Number of clusters to create (default: 5)
        embedding_generator: Optional EmbeddingGenerator instance (unused, for compatibility)
    
    Returns:
        Dict mapping flashcard_id to cluster_id
    
    Raises:
        ImportError: If scikit-learn is not installed
    
    Example:
        >>> clusters = cluster_flashcards("offline_ai.db", n_clusters=5)
        >>> print(f"Flashcard 1 is in cluster {clusters[1]}")
    """
    if KMeans is None:
        raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Load flashcards with embeddings
        flashcard_query = conn.execute(
            """SELECT f.id, f.embedding_id 
               FROM flashcards f 
               WHERE f.embedding_id IS NOT NULL"""
        ).fetchall()
        
        if len(flashcard_query) < n_clusters:
            logger.warning(f"Only {len(flashcard_query)} flashcards found, reducing clusters to match")
            n_clusters = max(1, len(flashcard_query))
        
        flashcard_ids = []
        embedding_matrix = []
        
        for fc_id, emb_id in flashcard_query:
            emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (emb_id,)).fetchone()
            if emb_row and emb_row[0]:
                flashcard_ids.append(fc_id)
                embedding_matrix.append(np.frombuffer(emb_row[0], dtype=np.float32))
        
        if not embedding_matrix:
            logger.warning("No flashcard embeddings found for clustering")
            return {}
        
        # Apply K-means clustering
        X = np.vstack(embedding_matrix)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create mapping
        clusters = {fc_id: int(cluster_id) for fc_id, cluster_id in zip(flashcard_ids, cluster_labels)}
        
        logger.info(f"Clustered {len(clusters)} flashcards into {n_clusters} groups")
        return clusters
        
    except Exception as e:
        logger.error(f"Failed to cluster flashcards: {e}", exc_info=True)
        raise
    finally:
        conn.close()


def find_similar_flashcards_for_node(
    db_path: str,
    node_label: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Find most similar flashcards for a given knowledge graph node.
    
    Uses cosine similarity between node embedding and flashcard embeddings.
    
    Args:
        db_path: Path to SQLite database
        node_label: Label of the node to find flashcards for
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of dicts with keys:
            - flashcard_id: ID of the flashcard
            - question: Flashcard question text
            - answer: Flashcard answer text
            - similarity: Cosine similarity score
            - topic: Flashcard topic
    
    Example:
        >>> results = find_similar_flashcards_for_node("offline_ai.db", "photosynthesis", top_k=3)
        >>> for fc in results:
        >>>     print(f"{fc['question']} (similarity: {fc['similarity']:.3f})")
    """
    conn = sqlite3.connect(db_path)
    
    try:
        from utils.embeddings import cosine_similarity
        
        # Lookup node
        node_query = conn.execute(
            "SELECT id, embedding_id FROM nodes WHERE label = ? LIMIT 1",
            (node_label,)
        ).fetchone()
        
        if not node_query:
            logger.warning(f"Node not found: {node_label}")
            return []
        
        node_id, node_emb_id = node_query
        
        if not node_emb_id:
            logger.warning(f"Node {node_label} has no embedding")
            return []
        
        # Fetch node embedding
        node_emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (node_emb_id,)).fetchone()
        if not node_emb_row or not node_emb_row[0]:
            logger.warning(f"Embedding not found for node {node_label}")
            return []
        
        node_emb = np.frombuffer(node_emb_row[0], dtype=np.float32)
        
        # Load all flashcards with embeddings
        flashcard_query = conn.execute(
            """SELECT f.id, f.question, f.answer, f.context AS topic, f.embedding_id 
               FROM flashcards f 
               WHERE f.embedding_id IS NOT NULL"""
        ).fetchall()
        
        results = []
        for fc_id, question, answer, topic, emb_id in flashcard_query:
            emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (emb_id,)).fetchone()
            if emb_row and emb_row[0]:
                fc_emb = np.frombuffer(emb_row[0], dtype=np.float32)
                similarity = cosine_similarity(node_emb.tolist(), fc_emb.tolist())
                
                results.append({
                    'flashcard_id': fc_id,
                    'question': question,
                    'answer': answer,
                    'similarity': similarity,
                    'topic': topic
                })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Failed to find similar flashcards: {e}", exc_info=True)
        raise
    finally:
        conn.close()


def find_similar_nodes_for_flashcard(
    db_path: str,
    flashcard_id: int,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Find most similar knowledge graph nodes for a given flashcard.
    
    Uses cosine similarity between flashcard embedding and node embeddings.
    
    Args:
        db_path: Path to SQLite database
        flashcard_id: ID of the flashcard
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of dicts with keys:
            - node_id: ID of the node
            - label: Node label
            - node_type: Type of node (Concept, Topic, Entity)
            - similarity: Cosine similarity score
    
    Example:
        >>> results = find_similar_nodes_for_flashcard("offline_ai.db", flashcard_id=1, top_k=3)
        >>> for node in results:
        >>>     print(f"{node['label']} ({node['node_type']}) - similarity: {node['similarity']:.3f}")
    """
    conn = sqlite3.connect(db_path)
    
    try:
        from utils.embeddings import cosine_similarity
        
        # Lookup flashcard
        fc_query = conn.execute(
            "SELECT embedding_id FROM flashcards WHERE id = ? LIMIT 1",
            (flashcard_id,)
        ).fetchone()
        
        if not fc_query:
            logger.warning(f"Flashcard not found: {flashcard_id}")
            return []
        
        fc_emb_id = fc_query[0]
        
        if not fc_emb_id:
            logger.warning(f"Flashcard {flashcard_id} has no embedding")
            return []
        
        # Fetch flashcard embedding
        fc_emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (fc_emb_id,)).fetchone()
        if not fc_emb_row or not fc_emb_row[0]:
            logger.warning(f"Embedding not found for flashcard {flashcard_id}")
            return []
        
        fc_emb = np.frombuffer(fc_emb_row[0], dtype=np.float32)
        
        # Load all KG nodes with embeddings
        node_query = conn.execute(
            """SELECT n.id, n.label, n.node_type, n.embedding_id 
               FROM nodes n 
               WHERE n.node_type IN ('Concept', 'Topic', 'Entity') 
               AND n.embedding_id IS NOT NULL"""
        ).fetchall()
        
        results = []
        for node_id, label, node_type, emb_id in node_query:
            emb_row = conn.execute("SELECT vector FROM embeddings WHERE id = ?", (emb_id,)).fetchone()
            if emb_row and emb_row[0]:
                node_emb = np.frombuffer(emb_row[0], dtype=np.float32)
                similarity = cosine_similarity(fc_emb.tolist(), node_emb.tolist())
                
                results.append({
                    'node_id': node_id,
                    'label': label,
                    'node_type': node_type,
                    'similarity': similarity
                })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
        
    except Exception as e:
        logger.error(f"Failed to find similar nodes: {e}", exc_info=True)
        raise
    finally:
        conn.close()
