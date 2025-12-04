import json
from unittest.mock import MagicMock

MOCK_PARSED_CONTENT_RESPONSE = json.dumps({
    'parsed_content': {
        'topics': [{'id': 't1', 'title': 'Photosynthesis', 'subtopics': []}],
        'formulas': [{'id': 'f1', 'latex': 'E=mc^2', 'description': 'energy=mass'}],
        'concepts': [{'id': 'c1', 'name': 'Chlorophyll', 'definition': 'pigment'}],
        'questions': [{'id': 'q1', 'text': 'What is photosynthesis?', 'difficulty': 2}]
    }
})

MOCK_SUMMARY_RESPONSE = json.dumps({'brief': 'Short summary', 'detailed': 'Detailed summary with formulas E=mc^2'})
MOCK_FLASHCARD_RESPONSE = json.dumps([{'question': 'Q1', 'answer': 'A1', 'difficulty': 2}])
MOCK_QUIZ_RESPONSE = json.dumps({'questions': [{'text': 'Q1', 'options': ['A','B'], 'correct': 'A'}]})
MOCK_MINDMAP_RESPONSE = json.dumps({'nodes': [{'id': 'n1', 'label': 'Root', 'children': []}], 'edges': []})

def mock_openai_create(*args, **kwargs):
    # determine response by prompt contents
    prompt = ''
    if 'messages' in kwargs:
        msgs = kwargs['messages']
        prompt = ' '.join([m.get('content', '') for m in msgs if isinstance(m, dict)])

    def build_resp(obj):
        return {
            'id': 'mock-1',
            'object': 'chat.completion',
            'created': 0,
            'model': kwargs.get('model', 'mock-model'),
            'choices': [{'message': obj}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20}
        }

    # For parsing, return a function_call style response with arguments string
    if 'parse' in prompt or 'parsed_content' in prompt:
        message = {'role': 'assistant', 'function_call': {'name': 'parsed_content', 'arguments': MOCK_PARSED_CONTENT_RESPONSE}}
        return build_resp(message)
    if 'summar' in prompt:
        message = {'role': 'assistant', 'content': MOCK_SUMMARY_RESPONSE}
        return build_resp(message)
    if 'flashcard' in prompt:
        message = {'role': 'assistant', 'content': MOCK_FLASHCARD_RESPONSE}
        return build_resp(message)
    if 'quiz' in prompt:
        message = {'role': 'assistant', 'content': MOCK_QUIZ_RESPONSE}
        return build_resp(message)
    if 'mindmap' in prompt:
        message = {'role': 'assistant', 'content': MOCK_MINDMAP_RESPONSE}
        return build_resp(message)
    # default
    message = {'role': 'assistant', 'content': MOCK_PARSED_CONTENT_RESPONSE}
    return build_resp(message)
