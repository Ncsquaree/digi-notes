from modules.semantic.summarizer import generate_summary


def test_generate_summary_brief(mock_openai_client, mock_redis_client, sample_parsed_content):
    res = generate_summary(sample_parsed_content, mode='brief')
    assert isinstance(res, dict)
    # brief may be None depending on mock; ensure metadata present
    assert 'metadata' in res
