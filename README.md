# Tokusan

Tokusan is an AI-powered Python module that can explain the output and interpretation of natural language processing models using Japanese text data.

## Overview

Tokusan extends LIME (Local Interpretable Model-Agnostic Explanations) with robust support for Japanese text processing and plain language explanation generation in both English and Japanese.

## Installation

```bash
# Install the package
pip install -e .

# For Japanese text support
pip install sudachipy sudachidict_core
```

## Quick Start

```python
from tokusan import TextExplainer, print_lime_narrative_jp

# Create explainer for Japanese text
explainer = TextExplainer(
    class_names=['フェイク', '本物'],
    lang='jp'
)

# Explain a prediction
exp = explainer.explain_instance(
    text="このニュースは信頼できる内容です",
    classifier_fn=model.predict_proba
)

# Get explanation as word-weight pairs
print(exp.as_list(label=1))

# Get Japanese plain language summary
print_lime_narrative_jp(exp)
```

## Features

- **Japanese Tokenization**: Automatic word segmentation using SudachiPy
- **Plain Language Explanations**: Human-readable explanations in English and Japanese
- **Model-Agnostic**: Works with any text classifier (sklearn, neural networks, etc.)
- **Visualization**: Generate bar charts and HTML visualizations of explanations

## License

BSD-2-Clause
