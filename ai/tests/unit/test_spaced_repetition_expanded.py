from modules.flashcards.spaced_repetition import FlashcardGenerator, generate_flashcards


def test_map_text_difficulty_and_generation(monkeypatch, sample_parsed_content, mock_openai_client):
    # ensure generator instance can be created (env from .env.test should provide OPENAI_API_KEY)
    g = FlashcardGenerator.get_instance()
    # test difficulty mapping via exposed method name (private method)
    d_easy = g._map_text_difficulty('easy')
    d_med = g._map_text_difficulty('medium')
    d_hard = g._map_text_difficulty('hard')
    assert d_easy <= d_med <= d_hard

    # generation using parsed content (mock OpenAI provides flashcard JSON)
    res = generate_flashcards(sample_parsed_content(), count=3)
    assert isinstance(res, dict)
    assert 'flashcards' in res
    assert isinstance(res['flashcards'], list)
