import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from sklearn import metrics

import models
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100)
args = parser.parse_args()



# ---------------- SPLIT ----------------
def create_split(df, ratio=0.2):
    drugs = list(set(df['d1']).union(set(df['d2'])))
    random.shuffle(drugs)

    split = int(len(drugs) * (1 - ratio))
    old = set(drugs[:split])
    new = set(drugs[split:])

    train, s1, s2 = [], [], []

    for _, row in df.iterrows():
        d1, d2, r = row['d1'], row['d2'], row['type']

        if d1 in old and d2 in old:
            train.append((d1, d2, r))
        elif d1 in new and d2 in new:
            s1.append((d1, d2, r))
        else:
            s2.append((d1, d2, r))

    print(f"Train: {len(train)}")
    print(f"S1 (new-new): {len(s1)}")
    print(f"S2 (new-old): {len(s2)}")

    return train, s1, s2


def split_train_val(data, val_ratio=0.1):
    random.shuffle(data)
    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]


# ---------------- METRICS ----------------
def compute_metrics(pred, gt):
    pred_bin = (pred >= 0.5).astype(int)

    acc = metrics.accuracy_score(gt, pred_bin)
    auc = metrics.roc_auc_score(gt, pred)
    auprc = metrics.average_precision_score(gt, pred)
    f1 = metrics.f1_score(gt, pred_bin)

    return acc, auc, auprc, f1


# ---------------- BATCH ----------------
def compute_batch(batch, model, device):
    pos, neg = batch

    pos = [x.to(device) for x in pos]
    neg = [x.to(device) for x in neg]

    p_score = model(pos)
    n_score = model(neg)

    prob = torch.sigmoid(torch.cat([p_score, n_score])).detach().cpu().numpy()
    gt = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])

    return p_score, n_score, prob, gt


# ---------------- TRAIN ----------------
def train(model, train_loader, val_loader, device, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    best_auc = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_pred, train_gt = [], []

        for batch in train_loader:
            p_score, n_score, prob, gt = compute_batch(batch, model, device)

            loss = -(torch.log(torch.sigmoid(p_score)).mean() +
                     torch.log(1 - torch.sigmoid(n_score)).mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.append(prob)
            train_gt.append(gt)

        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)

        train_acc, train_auc, train_auprc, train_f1 = compute_metrics(train_pred, train_gt)

        # -------- VALIDATION --------
        model.eval()
        val_pred, val_gt = [], []

        with torch.no_grad():
            for batch in val_loader:
                _, _, prob, gt = compute_batch(batch, model, device)
                val_pred.append(prob)
                val_gt.append(gt)

        val_pred = np.concatenate(val_pred)
        val_gt = np.concatenate(val_gt)

        val_acc, val_auc, val_auprc, val_f1 = compute_metrics(val_pred, val_gt)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"\nEpoch {epoch}")
        print(f"Train -> Acc: {train_acc:.4f} | AUC: {train_auc:.4f} | AUPRC: {train_auprc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   -> Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | AUPRC: {val_auprc:.4f} | F1: {val_f1:.4f}")

    print("Best Val AUC:", best_auc)


# ---------------- TEST ----------------
def test(model, loader, device, name):
    model.eval()
    pred, gt = [], []

    with torch.no_grad():
        for batch in loader:
            _, _, prob, g = compute_batch(batch, model, device)
            pred.append(prob)
            gt.append(g)

    pred = np.concatenate(pred)
    gt = np.concatenate(gt)

    acc, auc, auprc, f1 = compute_metrics(pred, gt)

    print(f"\n{name}")
    print("Acc:", round(acc,4))
    print("AUC:", round(auc,4))
    print("AUPRC:", round(auprc,4))
    print("F1:", round(f1,4))


# ---------------- MAIN ----------------
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = pd.read_csv("data/ddis.csv")

    train_data, test_s1, test_s2 = create_split(df)
    train_data, val_data = split_train_val(train_data)

    train_ds = DrugDataset(train_data, neg_ent=1)
    val_ds   = DrugDataset(val_data, disjoint_split=False)
    s1_ds    = DrugDataset(test_s1, disjoint_split=False)
    s2_ds    = DrugDataset(test_s2, disjoint_split=False)

    train_loader = DrugDataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DrugDataLoader(val_ds, batch_size=256)
    s1_loader    = DrugDataLoader(s1_ds, batch_size=256)
    s2_loader    = DrugDataLoader(s2_ds, batch_size=256)

    model = models.MASMDDI(TOTAL_ATOM_FEATS, 256, 86).to(device)

    print("Training started...")
    train(model, train_loader, val_loader, device, args.n_epochs)

    model.load_state_dict(torch.load("best_model.pth"))

    test(model, s1_loader, device, "S1 (NEW-NEW)")
    test(model, s2_loader, device, "S2 (NEW-OLD)")