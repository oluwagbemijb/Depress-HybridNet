import torch
import numpy as np
from models import DepressHybridNet
from dataset import DepressionDataset, collate_fn
from utils import load_config, get_device, compute_metrics
from torch.utils.data import DataLoader

def evaluate(checkpoint_path=None, split_csv=None):
    cfg = load_config("config.yaml")
    device = get_device(cfg)
    if checkpoint_path is None:
        checkpoint_path = cfg["paths"].get("checkpoint", "runs/checkpoint.pt")
    if split_csv is None:
        split_csv = "data/splits/test.csv"

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DepressHybridNet(ckpt.get("cfg", cfg))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    ds = DepressionDataset(split_csv, tokenizer_name=cfg["model"]["bert_model_name"], max_length=cfg["model"]["max_seq_len"])
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate_fn)

    ys, yps = [], []
    attn_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            behavior = batch["behavior"].to(device)
            labels = batch["labels"].to(device)

            logits, probs, attn = model(input_ids, attention_mask, behavior)
            ys.append(labels.cpu().numpy())
            yps.append(probs.cpu().numpy())
            attn_list.append(attn.cpu().numpy())

    ys = np.concatenate(ys)
    yps = np.concatenate(yps)
    attn_arr = np.concatenate(attn_list, axis=0)
    metrics = compute_metrics(ys, yps)
    print("Evaluation metrics:", metrics)
    # return metrics and attention for further inspection
    return metrics, attn_arr

if __name__ == "__main__":
    evaluate()
