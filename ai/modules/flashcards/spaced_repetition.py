"""Simple SM-2 spaced repetition implementation (placeholder).
Refer to the SM-2 algorithm for production use.
"""
from datetime import datetime, timedelta

def initial_card():
    return { 'interval': 1, 'repetitions': 0, 'ef': 2.5, 'due': datetime.utcnow() }

def update_card(card, quality: int):
    # quality: 0-5
    if quality < 3:
        card['repetitions'] = 0
        card['interval'] = 1
    else:
        card['repetitions'] += 1
        if card['repetitions'] == 1:
            card['interval'] = 1
        elif card['repetitions'] == 2:
            card['interval'] = 6
        else:
            card['interval'] = round(card['interval'] * card['ef'])
        # update ease factor
        card['ef'] = max(1.3, card['ef'] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    card['due'] = datetime.utcnow() + timedelta(days=card['interval'])
    return card
