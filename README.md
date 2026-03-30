# nlp_product_reviews
Use DeBERTa-v3-base to train a classifier for flagging unsafe product reviews.

## Tutorial
Based on the tutorial by [Jay Alammar](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/).

## Setup

Requires **Python 3.12+**.

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

On a GPU machine with CUDA 12.8, install with GPU extras:

```bash
pip install ".[gpu]"
```

## Configuration

All training hyperparameters are in `config.py`. Hardware is auto-detected:

| Hardware | Batch size | Accumulation | FP16 |
|----------|-----------|--------------|------|
| CUDA GPU (32GB) | 32 | 1 | Yes |
| CUDA GPU (16GB) | 8 | 4 | Yes |
| Apple MPS | 8 | 4 | No |
| CPU only | 4 | 8 | No |

Key settings:
- `NUM_EXAMPLES` — number of training samples (default: 3200)
- `NUM_EPOCHS` — training epochs (default: 8)
- `CLASS_1_WEIGHT` — weight for unsafe class loss (default: 8.0)
- `DECISION_THRESHOLD` — inference threshold for flagging (default: 0.3)

## Training

```bash
source venv/bin/activate
python classifier/bert_model_training.py
```

To run in the background with tmux:

```bash
tmux new -s training 'source venv/bin/activate && python classifier/bert_model_training.py'
```

Reattach with `tmux attach -t training`.

## Results

All outputs are saved under `./results/`:

- `training_log.txt` — training log
- `checkpoint-*/` — per-epoch checkpoints
- `model/microsoft_deberta-v3-base/` — saved model weights
- `microsoft_deberta-v3-base/<timestamp>_eval_result.json` — evaluation metrics

### Example output (3200 samples, 8 epochs, MPS)

| Epoch | Accuracy | AUC   | Recall (class 1) | F1    |
|-------|----------|-------|-------------------|-------|
| 1     | 0.850    | 0.640 | 0.031             | 0.787 |
| 3     | 0.819    | 0.825 | 0.694             | 0.834 |
| 5     | 0.838    | 0.864 | 0.704             | 0.849 |
| 8     | 0.870    | 0.870 | 0.694             | 0.876 |
