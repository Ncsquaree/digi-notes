from modules.flashcards.spaced_repetition import generate_flashcards


def test_generate_flashcards_from_parsed(sample_parsed_content):
    res = generate_flashcards(sample_parsed_content, count=2)
    assert isinstance(res, dict)
    assert 'flashcards' in res
    assert isinstance(res['flashcards'], list)
