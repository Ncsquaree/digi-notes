"""
Pytest fixtures and test data for offline-ai tests.
"""
import pytest
from pathlib import Path


@pytest.fixture
def sample_text():
    """Academic text with headings, lists, and formulas."""
    return """# Photosynthesis

Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. The process occurs in two main stages:

1. Light-dependent reactions: These occur in the thylakoid membranes and produce ATP and NADPH.
2. Light-independent reactions (Calvin cycle): These occur in the stroma and use ATP and NADPH to fix CO2 into glucose.

Key Formula:
6CO2 + 6H2O + light energy → C6H12O6 + 6O2

Important Concepts:
- Chloroplast: Organelle where photosynthesis occurs
- Chlorophyll: Green pigment that absorbs light
- Stomata: Pores for gas exchange

The Photosynthesis Equation:
Plants absorb CO2 from the atmosphere and water from the soil. Using light energy, they convert these into glucose and oxygen. This is crucial for life on Earth."""


@pytest.fixture
def empty_text():
    """Empty and whitespace-only text."""
    return {
        'empty': '',
        'spaces': '   ',
        'newlines': '\n\n\n',
        'tabs': '\t\t\t'
    }


@pytest.fixture
def markdown_text():
    """Text with markdown syntax."""
    return """# Introduction

This is **bold text** and *italic text*.

## Subsection

Here is a [link](https://example.com).

```python
def hello():
    print("code block")
```

- Bullet list
- Item 2
  - Nested item

1. Numbered list
2. Second item"""


@pytest.fixture
def long_paragraph():
    """Single paragraph longer than 500 characters."""
    return """The mitochondria is often referred to as the powerhouse of the cell because it is the site of aerobic cellular respiration, a metabolic process that breaks down glucose and produces ATP, the energy currency of the cell. The mitochondrion has a double membrane structure with the outer membrane being smooth while the inner membrane contains infoldings called cristae that increase the surface area for biochemical reactions. Inside the mitochondria, the citric acid cycle and the electron transport chain work together to extract energy from nutrients and generate ATP molecules that fuel cellular processes. This process is so efficient that one glucose molecule can produce up to 36 ATP molecules, compared to just 2 ATP molecules produced during anaerobic glycolysis. Different cell types have different numbers of mitochondria depending on their energy requirements; for example, muscle cells have many more mitochondria than adipose tissue because they require large amounts of ATP for muscle contraction."""


@pytest.fixture
def formula_text():
    """Text with mathematical equations and formulas."""
    return """Physics Formulas

Newton's Second Law:
F = ma

Kinetic Energy:
KE = ½mv²

The quadratic formula for solving ax² + bx + c = 0 is:
x = (-b ± √(b² - 4ac)) / 2a

Chemical Formula:
H2SO4 + 2NaOH → Na2SO4 + 2H2O

Einstein's Mass-Energy Equivalence:
E = mc²"""


@pytest.fixture
def complex_pdf_text():
    """Text extracted from PDF with common artifacts."""
    return """Page 1

Introduction to Biology
Dr. Smith

Introduction

Biology is the study of life and living organisms. It encompasses
a wide range of topics from molecular structures to ecosystem dynamics.

Research Methods

The scientific method provides a framework for conducting research:
1- Observation of phenomena
2- Hypothesis formation
3- Experimental design
4- Data collection and analysis

Page 2

Results

Our findings show significant correlations between temperature and enzyme
activity. The reaction rate doubled with each 10°C increase up to 40°C,
after which it declined rapidly.

Discussion

These results support the theory of enzyme kinetics. The optimal tem-
perature for this enzyme is approximately 37°C, which aligns with mamma-
lian body temperature.

Conclusions

Further research is needed to determine long-term effects."""


@pytest.fixture
def sample_chunks():
    """Pre-processed chunks with metadata for Phase 5 testing."""
    return [
        {
            'text': 'Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.',
            'metadata': {
                'topic': 'Photosynthesis',
                'type': 'paragraph',
                'academic_level': 'basic',
                'has_formula': False,
                'entities': [{'text': 'Photosynthesis', 'type': 'Concept', 'confidence': 0.9}],
                'embedding_id': 1
            }
        },
        {
            'text': 'The Calvin cycle uses ATP and NADPH to convert CO2 into glucose through a series of enzyme-catalyzed reactions.',
            'metadata': {
                'topic': 'Calvin Cycle',
                'type': 'paragraph',
                'academic_level': 'intermediate',
                'has_formula': True,
                'entities': [
                    {'text': 'Calvin cycle', 'type': 'Concept', 'confidence': 0.95},
                    {'text': 'ATP', 'type': 'Concept', 'confidence': 0.9},
                    {'text': 'NADPH', 'type': 'Concept', 'confidence': 0.9}
                ],
                'embedding_id': 2
            }
        },
        {
            'text': 'The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH needed for the Calvin cycle.',
            'metadata': {
                'topic': 'Light Reactions',
                'type': 'paragraph',
                'academic_level': 'advanced',
                'has_formula': False,
                'entities': [
                    {'text': 'light-dependent reactions', 'type': 'Concept', 'confidence': 0.92},
                    {'text': 'thylakoid', 'type': 'Concept', 'confidence': 0.88}
                ],
                'embedding_id': 3
            }
        }
    ]


@pytest.fixture
def mock_ner_extractor():
    """Mock NER extractor for Phase 5 testing."""
    class MockNER:
        def extract_entities(self, text):
            return []
        
        def extract_from_chunks(self, chunks):
            return chunks  # Return chunks as-is
    
    return MockNER()
