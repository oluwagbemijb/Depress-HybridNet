# Depress-HybridNet - Minimal Implementation

Files:
- `models.py` - model definitions
- `dataset.py` - dataset & dataloader
- `train.py` - train loop with early stopping and checkpointing
- `eval.py` - evaluation script
- `utils.py` - config, seed, metrics helpers
- `config.yaml` - configuration

Usage:
1. Prepare a CSV `data/dataset.csv` with columns: `text`, `label` (0/1), plus optional numeric behavioral columns.
2. Edit `config.yaml` paths if needed.
3. `pip install -r requirements.txt`
4. `python train.py`
5. `python eval.py`
