"""Graph builder stub — converts structured content into nodes/edges."""
from typing import Dict, Any, List

def build_graph(structured: Dict[str, Any]) -> Dict[str, Any]:
	# Placeholder that returns a simple node/edge list
	nodes = []
	edges = []
	topic = structured.get('topic')
	if topic:
		nodes.append({'id': topic, 'type': 'topic'})
	for i, sub in enumerate(structured.get('subtopics', [])):
		nodes.append({'id': sub, 'type': 'subtopic'})
		edges.append({'from': topic, 'to': sub, 'relation': 'has_subtopic'})
	return {'nodes': nodes, 'edges': edges}


if __name__ == '__main__':
	sample = { 'topic': 'Photosynthesis', 'subtopics': ['Light Reaction','Dark Reaction'] }
	print(build_graph(sample))

