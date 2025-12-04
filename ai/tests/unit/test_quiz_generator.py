from modules.semantic.quiz_generator import generate_quiz


def test_generate_quiz_basic(mock_openai_client, sample_parsed_content):
    res = generate_quiz(sample_parsed_content, question_count=2)
    assert isinstance(res, dict)
    assert 'questions' in res
    assert isinstance(res.get('questions'), list)
