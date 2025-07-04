# Repo guidelines for model-training scripts

This repository contains scripts for training transformer-based classifiers on product reviews.

## Ignore these files
The following files are experimental and should be ignored when modifying or analysing the code. Avoid referencing or editing them unless explicitly instructed.

- `bert_albert.py`
- `batch.py`
- `tfDataLoader.py`
- `test.py`

## Style
- Follow **PEP8** (4-space indentation, max 79 characters per line).
- Use docstrings for all public functions.

## Testing
There is no full test suite, but syntax errors can be caught by running:

```bash
python -m py_compile $(git ls-files '*.py' | grep -v 'test.py')
```

Run this command after modifying Python files (excluding `test.py`) and before committing.

