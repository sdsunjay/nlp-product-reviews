# nlp_product_reviews
Use BERT to pretrain a model for categorizing reviews

## tutorial
I am following the tutorial shown [here](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) by [Jay Alammar](https://twitter.com/JayAlammar)

## Requirements
This project requires **Python 3.12** and the GPU build of **PyTorch 2.7**
compiled with **CUDA 12.8**. Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

## Training Output

Here's an *small* example of the output from the BERT model training on my MacBook Pro (6 - Core Intel i7) on 1600 samples:
```
***** Running training *****
  Num examples = 1600
  Num Epochs = 1
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 1
  Total optimization steps = 100
  Number of trainable parameters = 109483778
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [45:33<00:00, 28.30s/it]
```

```
{'train_runtime': 2733.2875, 'train_samples_per_second': 0.585, 'train_steps_per_second': 0.037, 'train_loss': 0.5109413528442383, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [45:33<00:00, 27.33s/it]
***** Running Evaluation *****
  Num examples = 400
  Batch size = 64
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [02:10<00:00, 18.60s/it]
{'eval_loss': 0.4243595004081726, 'eval_accuracy': 0.8425, 'eval_runtime': 154.9798, 'eval_samples_per_second': 2.581, 'eval_steps_per_second': 0.045, 'epoch': 1.0}
```
