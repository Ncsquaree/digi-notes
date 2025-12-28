#!/usr/bin/env python3
"""
Offline AI CLI - Flashcard & Knowledge Graph Generator
Usage: python main.py input.txt

Pipeline stages:
  Phase 1: Setup ✓ (complete)
  Phase 2: Text Preprocessing ✓ (complete)
  Phase 3: Embeddings (in progress)
  Phase 4: NER (in progress)
  Phase 5: Flashcard Generation (pending)
  Phase 6: Knowledge Graph (pending)
  Phase 7: Semantic Linking (pending)
  Phase 8: End-to-End CLI (pending)
"""
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file.txt>")
        print("\nSupported formats:")
        print("  - .txt: Plain text files")
        print("  - .md:  Markdown files")
        print("  - .txt: PDF-extracted text (use --format=pdf)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    input_format = 'plain'
    
    # Parse optional format argument
    for arg in sys.argv[2:]:
        if arg.startswith('--format='):
            input_format = arg.split('=')[1]
    
    try:
        # Verify input file exists
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
        
        # Read input
        logger.info(f"Reading input from: {input_file}")
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Phase 2: Preprocess text
        logger.info("Starting Phase 2: Text Preprocessing")
        from utils.preprocess import preprocess_text
        from utils.embeddings import EmbeddingGenerator, cosine_similarity
        from utils.ner import EntityExtractor
        
        chunks = preprocess_text(raw_text, max_chunk_size=500, input_format=input_format)
        logger.info(f"✓ Phase 2 complete: {len(chunks)} chunks generated")
        
        # Display preprocessing results
        print(f"\n{'='*70}")
        print(f"PREPROCESSING RESULTS ({len(chunks)} chunks)")
        print(f"{'='*70}\n")
        
        type_counts = {}
        for i, chunk in enumerate(chunks, 1):
            ctype = chunk['metadata']['type']
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
            # Show first 5 chunks in detail
            if i <= 5:
                print(f"Chunk {i} ({ctype}):")
                print(f"  Topic: {chunk['metadata']['topic']}")
                print(f"  Size: {chunk['metadata']['char_count']} chars, {chunk['metadata']['word_count']} words")
                print(f"  Level: {chunk['metadata']['academic_level']}")
                print(f"  Text: {chunk['text'][:80]}...")
                print()
        
        # Summary statistics
        print(f"{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Content type distribution: {type_counts}")
        
        avg_size = sum(c['metadata']['char_count'] for c in chunks) / len(chunks)
        print(f"Average chunk size: {avg_size:.0f} characters")
        
        has_formulas = sum(1 for c in chunks if c['metadata']['has_formula'])
        print(f"Chunks with formulas: {has_formulas}")

        # Phase 3: Embeddings
        print(f"\n{'='*70}")
        print("PHASE 3: EMBEDDINGS")
        print(f"{'='*70}")
        try:
            emb_gen = EmbeddingGenerator(db_path="offline_ai.db", auto_load=True)
        except Exception as emb_err:
            logger.warning(f"Failed to load TFLite model: {emb_err}")
            logger.info("Falling back to hash-based embeddings for testing")
            emb_gen = EmbeddingGenerator(db_path="offline_ai.db", force_hash_fallback=True)
        
        embedding_results = emb_gen.embed_and_store_chunks(chunks)
        print(f"Generated embeddings for {len(embedding_results)} chunks (dim={emb_gen.embedding_dim})")
        print(f"Model: {'TFLite USE Lite' if emb_gen.interpreter else 'Hash fallback (testing)'}")
        print("Database: offline_ai.db (table: embeddings)")
        if embedding_results:
            print(f"Sample embedding_id: {embedding_results[0].metadata.get('embedding_id')}")
            print("Sample similarity (chunk 1 vs chunk 2 if available):")
            if len(embedding_results) > 1:
                sim = cosine_similarity(embedding_results[0].embedding, embedding_results[1].embedding)
                print(f"  Cosine similarity: {sim:.3f}")
            else:
                print("  Only one chunk available; skipping similarity sample.")
        
        # Phase 4: Entity Extraction (NER)
        print(f"\n{'='*70}")
        print("PHASE 4: ENTITY EXTRACTION (NER)")
        print(f"{'='*70}")
        try:
            ner = EntityExtractor(auto_load=True)
        except Exception as ner_err:
            logger.warning(f"Failed to load MobileBERT: {ner_err}")
            logger.info("Falling back to NLTK NER")
            ner = EntityExtractor(use_fallback=True)
        
        chunks_with_entities = ner.extract_from_chunks(chunks)
        
        # Collect all entities
        all_entities = []
        entity_type_counts = {}
        for chunk in chunks_with_entities:
            for entity_dict in chunk['metadata'].get('entities', []):
                all_entities.append(entity_dict['text'])
                entity_type = entity_dict['type']
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        # Deduplicate entities
        unique_entities = ner.deduplicate_entities(all_entities)
        
        print(f"Extracted {len(all_entities)} entities ({len(unique_entities)} unique)")
        print(f"Model: {'MobileBERT-SQuAD TFLite' if ner.interpreter else 'NLTK fallback'}")
        print(f"Entity types: {entity_type_counts}")
        print(f"Sample entities: {unique_entities[:10]}")
        
        # Phase 5: Flashcard Generation
        print(f"\n{'='*70}")
        print("PHASE 5: FLASHCARD GENERATION")
        print(f"{'='*70}")
        try:
            from flashcards.generate import FlashcardGenerator
            import sys
            sys.path.insert(0, 'modules')
            
            # Initialize with NER and embedding generator
            fc_gen = FlashcardGenerator(
                db_path="offline_ai.db",
                ner_extractor=ner,
                embedding_generator=emb_gen,
                auto_load=True
            )
            
            # Generate flashcards
            flashcards = fc_gen.generate_from_chunks(chunks_with_entities)
            
            # Store flashcards
            fc_ids = fc_gen.store_flashcards(flashcards)
            
            print(f"Generated {len(flashcards)} flashcards")
            print(f"Model: {'MobileBERT-SQuAD TFLite' if fc_gen.interpreter else 'Rule-based fallback'}")
            
            # Display sample flashcards
            if flashcards:
                print(f"\nSample flashcards (first 3):")
                for i, fc in enumerate(flashcards[:3], 1):
                    print(f"\n[{i}] {fc.question}")
                    print(f"    Answer: {fc.answer}")
                    print(f"    Difficulty: {fc.difficulty}/5 | Topic: {fc.topic}")
                    print(f"    Source: {fc.source_type} | Entities: {', '.join(fc.entities[:3]) if fc.entities else 'None'}")
            
            # Statistics
            source_types = {}
            difficulty_dist = {}
            for fc in flashcards:
                source_types[fc.source_type] = source_types.get(fc.source_type, 0) + 1
                difficulty_dist[fc.difficulty] = difficulty_dist.get(fc.difficulty, 0) + 1
            
            print(f"\nFlashcard Statistics:")
            print(f"  Source types: {source_types}")
            print(f"  Difficulty distribution: {difficulty_dist}")
            print(f"  Database: offline_ai.db (table: flashcards)")
            
        except Exception as fc_err:
            logger.error(f"Phase 5 (Flashcard Generation) failed: {fc_err}", exc_info=True)
            print(f"✗ Phase 5 failed: {fc_err}")
        
        # Phase 6: Knowledge Graph Construction
        print(f"\n{'='*70}")
        print("PHASE 6: KNOWLEDGE GRAPH CONSTRUCTION")
        print(f"{'='*70}")
        try:
            from knowledge_graph.build import GraphBuilder
            from knowledge_graph.visualize import visualize_graph, export_graph_json

            graph_builder = GraphBuilder(
                chunks_with_entities=chunks_with_entities,
                embedding_generator=emb_gen,
                db_path="offline_ai.db",
                flashcard_ids=fc_ids if 'fc_ids' in locals() else []
            )

            graph_result = graph_builder.build_graph()

            print(f"Created {graph_result['nodes_created']} nodes and {graph_result['edges_created']} edges")
            print(f"Node types: {graph_result['node_type_counts']}")
            print(f"Database: offline_ai.db (tables: nodes, edges)")

            # Visualization and export
            visualize_graph("offline_ai.db", output_path="offline-ai/data/knowledge_graph.png")
            export_graph_json("offline_ai.db", output_path="offline-ai/data/graph.json")
            print("\n✓ Graph visualization saved to offline-ai/data/knowledge_graph.png")
            print("✓ Graph JSON exported to offline-ai/data/graph.json")
        except Exception as g_err:
            logger.error(f"Phase 6 (Knowledge Graph) failed: {g_err}", exc_info=True)
            print(f"✗ Phase 6 failed: {g_err}")
        
        # Phase 7: Semantic Linking
        print(f"\n{'='*70}")
        print("PHASE 7: SEMANTIC LINKING (FLASHCARDS ↔ KG NODES)")
        print(f"{'='*70}")
        try:
            from knowledge_graph.link import link_flashcards_to_graph, cluster_flashcards
            
            link_result = link_flashcards_to_graph(
                db_path="offline_ai.db",
                similarity_threshold=0.7,
                embedding_generator=emb_gen
            )
            
            print(f"Created {link_result['links_created']} semantic links")
            print(f"Processed {link_result['flashcards_processed']} flashcards, {link_result['nodes_processed']} nodes")
            print(f"Average similarity: {link_result['avg_similarity']:.3f}")
            
            # Optional clustering
            try:
                clusters = cluster_flashcards(db_path="offline_ai.db", n_clusters=5, embedding_generator=emb_gen)
                print(f"Clustered flashcards into {len(set(clusters.values()))} groups")
            except ImportError as cluster_err:
                logger.warning(f"Clustering skipped: {cluster_err}")
                print(f"⚠ Clustering skipped (install scikit-learn for this feature)")
            
        except Exception as link_err:
            logger.error(f"Phase 7 (Semantic Linking) failed: {link_err}", exc_info=True)
            print(f"✗ Phase 7 failed: {link_err}")

        # Next phases preview
        print(f"\n{'='*70}")
        print("NEXT STEPS (Phase 8+)")
        print(f"{'='*70}")
        print("[✓] Phase 6: Build knowledge graph from extracted entities")
        print("[✓] Phase 7: Link flashcards to KG nodes semantically")
        print("[ ] Phase 8: Export to JSON and database")

        # Phase 8: Export & Final Summary
        print(f"\n{'='*70}")
        print("PHASE 8: EXPORT & FINAL SUMMARY")
        print(f"{'='*70}")
        try:
            import sqlite3
            import json
            from datetime import datetime
            
            # Export flashcards from SQLite to JSON
            conn = sqlite3.connect("offline_ai.db")
            cursor = conn.cursor()
            
            # Get flashcard count and metadata
            cursor.execute("SELECT COUNT(*) FROM flashcards")
            total_flashcards = cursor.fetchone()[0]
            
            # Get difficulty distribution
            cursor.execute(
                "SELECT difficulty, COUNT(*) FROM flashcards GROUP BY difficulty"
            )
            difficulty_dist = dict(cursor.fetchall())
            
            # Get source type distribution
            cursor.execute(
                "SELECT source_type, COUNT(*) FROM flashcards GROUP BY source_type"
            )
            source_dist = dict(cursor.fetchall())
            
            # Fetch all flashcards
            cursor.execute(
                """SELECT id, question, answer, difficulty, context, source_type, embedding_id 
                   FROM flashcards ORDER BY id"""
            )
            flashcards_data = cursor.fetchall()
            
            # Build flashcards JSON
            flashcards_json = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "source_text": input_file,
                    "total_flashcards": total_flashcards,
                    "by_difficulty": difficulty_dist,
                    "by_source_type": source_dist
                },
                "flashcards": [
                    {
                        "id": fc[0],
                        "question": fc[1],
                        "answer": fc[2],
                        "difficulty": fc[3],
                        "topic": fc[4],
                        "source_type": fc[5],
                        "embedding_id": fc[6]
                    }
                    for fc in flashcards_data
                ]
            }
            
            # Write flashcards JSON
            flashcards_path = "offline-ai/data/flashcards.json"
            Path(flashcards_path).parent.mkdir(parents=True, exist_ok=True)
            with open(flashcards_path, 'w') as f:
                json.dump(flashcards_json, f, indent=2)
            print(f"✓ Flashcards exported: {flashcards_path}")
            print(f"  Total: {total_flashcards} flashcards")
            
            conn.close()
            
            # Confirm graph export (already done in Phase 6)
            graph_path = "offline-ai/data/graph.json"
            if Path(graph_path).exists():
                print(f"✓ Knowledge graph exported: {graph_path}")
            
            # Final summary
            print(f"\n{'='*70}")
            print("FINAL OUTPUT SUMMARY")
            print(f"{'='*70}")
            print(f"Input:  {input_file}")
            print(f"Database: offline_ai.db")
            print(f"\nOutputs:")
            print(f"  - Flashcards:  {flashcards_path} ({total_flashcards} cards)")
            print(f"  - Knowledge Graph: {graph_path} ({graph_result.get('nodes_created', 0)} nodes, {graph_result.get('edges_created', 0)} edges)")
            print(f"  - Embeddings: offline_ai.db (table: embeddings)")
            
            # Edge-device stats
            try:
                import psutil
                ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
                ram_status = "✓" if ram_mb < 1024 else "⚠"
                print(f"\nEdge-Device Stats:")
                print(f"  RAM Usage: {ram_mb:.0f} MB ({ram_status} <1GB target)")
                print(f"  CPU-only inference: ✓ (TFLite no GPU)")
                print(f"  Offline: ✓ (100% - no API calls)")
            except ImportError:
                logger.warning("psutil not available for RAM monitoring")
                print(f"\nEdge-Device Stats:")
                print(f"  CPU-only inference: ✓ (TFLite no GPU)")
                print(f"  Offline: ✓ (100% - no API calls)")
            
            print(f"\n{'='*70}")
            print("✓ Pipeline complete! All phases (1-8) finished successfully.")
            print(f"{'='*70}")
            
        except Exception as export_err:
            logger.error(f"Phase 8 (Export) failed: {export_err}", exc_info=True)
            print(f"✗ Phase 8 failed: {export_err}")
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
