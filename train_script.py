from datetime import datetime
import time
import argparse
import random

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS


# ===================== ARGUMENTS =====================
parser = argparse.ArgumentParser()

parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS)
parser.add_argument('--n_atom_hid', type=int, default=256)
parser.add_argument('--rel_total', type=int, default=86)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print("Using device:", device)


# ===================== S2 SPLIT FUNCTION =====================
def create_two_sides_split(triples, test_ratio=0.2, seed=42):
    random.seed(seed)

    drugs = list(set([h for h, _, _ in triples] + [t for _, t, _ in triples]))
    random.shuffle(drugs)

    split_idx = int(len(drugs) * (1 - test_ratio))

    train_drugs = set(drugs[:split_idx])
    test_drugs = set(drugs[split_idx:])

    train_data, test_data = [], []

    for h, t, r in triples:
        if h in train_drugs and t in train_drugs:
            train_data.append((h, t, r))
        elif h in test_drugs and t in test_drugs:
            test_data.append((h, t, r))

    return train_data, test_data


# ===================== LOAD DATA =====================
df_all = pd.read_csv('data/ddis.csv')

all_triples = list(zip(df_all['d1'], df_all['d2'], df_all['type']))

train_tup, test_tup = create_two_sides_split(all_triples)

# Split test → val + test
split = len(test_tup) // 2
val_tup = test_tup[:split]
test_tup = test_tup[split:]


# ===================== CHECK SPLIT =====================
train_drugs = set([h for h,_,_ in train_tup] + [t for _,t,_ in train_tup])
test_drugs = set([h for h,_,_ in test_tup] + [t for _,t,_ in test_tup])

print("Train drugs:", len(train_drugs))
print("Test drugs:", len(test_drugs))
print("Overlap:", len(train_drugs & test_drugs))  # MUST BE 0


# ===================== DATASET =====================
train_data = DrugDataset(train_tup, neg_ent=args.neg_samples)
val_data   = DrugDataset(val_tup, disjoint_split=False)
test_data  = DrugDataset(test_tup, disjoint_split=False)

train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader   = DrugDataLoader(val_data, batch_size=args.batch_size * 2)
test_loader  = DrugDataLoader(test_data, batch_size=args.batch_size * 2)


# ===================== METRICS =====================
def compute_metrics(pred, gt):
    pred_binary = (pred >= 0.5).astype(int)

    acc = metrics.accuracy_score(gt, pred_binary)
    auc = metrics.roc_auc_score(gt, pred)
    auprc = metrics.average_precision_score(gt, pred)
    f1 = metrics.f1_score(gt, pred_binary)

    return acc, auc, auprc, f1


def compute_batch(batch):
    pos, neg = batch

    pos = [x.to(device) for x in pos]
    neg = [x.to(device) for x in neg]

    p_score = model(pos)
    n_score = model(neg)

    probas = torch.sigmoid(torch.cat([p_score, n_score])).detach().cpu().numpy()
    labels = np.concatenate([
        np.ones(len(p_score)),
        np.zeros(len(n_score))
    ])

    return p_score, n_score, probas, labels


# ===================== TRAIN =====================
def train():
    best_val_auc = 0

    print("\nTraining started:", datetime.now())

    for epoch in range(1, args.n_epochs + 1):
        start = time.time()

        # ---- TRAIN ----
        model.train()
        total_loss = 0

        for batch in train_loader:
            p_score, n_score, _, _ = compute_batch(batch)

            loss, _, _ = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---- EVALUATE TRAIN ----
        model.eval()
        train_pred, train_gt = [], []

        with torch.no_grad():
            for batch in train_loader:
                _, _, probas, labels = compute_batch(batch)
                train_pred.append(probas)
                train_gt.append(labels)

        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)

        train_acc, train_auc, train_auprc, train_f1 = compute_metrics(train_pred, train_gt)

        # ---- VALIDATION ----
        val_pred, val_gt = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                p_score, n_score, probas, labels = compute_batch(batch)
                loss, _, _ = loss_fn(p_score, n_score)

                val_loss += loss.item()
                val_pred.append(probas)
                val_gt.append(labels)

        val_loss /= len(val_loader)

        val_pred = np.concatenate(val_pred)
        val_gt = np.concatenate(val_gt)

        val_acc, val_auc, val_auprc, val_f1 = compute_metrics(val_pred, val_gt)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

        scheduler.step()

        print(f"\nEpoch {epoch:03d} | Time: {time.time() - start:.2f}s")
        print(f"Train -> Loss: {train_loss:.4f} | "
              f"Acc: {train_acc:.4f} | "
              f"ROC-AUC: {train_auc:.4f} | "
              f"AUPRC: {train_auprc:.4f} | "
              f"F1: {train_f1:.4f}")

        print(f"Val   -> Loss: {val_loss:.4f} | "
              f"Acc: {val_acc:.4f} | "
              f"ROC-AUC: {val_auc:.4f} | "
              f"AUPRC: {val_auprc:.4f} | "
              f"F1: {val_f1:.4f}")

    print("\nBest Val AUC:", best_val_auc)


# ===================== TEST =====================
def test():
    print("\n===== TEST RESULTS (S2 SETTING) =====")

    model.eval()
    test_pred, test_gt = [], []

    with torch.no_grad():
        for batch in test_loader:
            _, _, probas, labels = compute_batch(batch)
            test_pred.append(probas)
            test_gt.append(labels)

    test_pred = np.concatenate(test_pred)
    test_gt = np.concatenate(test_gt)

    acc, auc, auprc, f1 = compute_metrics(test_pred, test_gt)

    print("Test Accuracy :", round(acc, 4))
    print("Test ROC-AUC  :", round(auc, 4))
    print("Test AUPRC    :", round(auprc, 4))
    print("Test F1-score :", round(f1, 4))


# ===================== MODEL =====================
model = models.MASMDDI(
    args.n_atom_feats,
    args.n_atom_hid,
    args.rel_total,
    args.num_layers
).to(device)

loss_fn = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.96 ** e)


# ===================== RUN =====================
train()

model.load_state_dict(torch.load("best_model.pth"))
test()