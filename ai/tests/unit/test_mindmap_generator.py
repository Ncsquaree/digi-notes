from modules.semantic.mindmap_generator import generate_mindmap
from tests.fixtures.sample_data import small_parsed_content


def test_generate_mindmap_basic(mock_openai_client):
    parsed = small_parsed_content()
    res = generate_mindmap(parsed)
    assert isinstance(res, dict)
    assert 'nodes' in res and 'edges' in res
