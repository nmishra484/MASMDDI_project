from datetime import datetime
import time
import argparse

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
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=4)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print("Using device:", device)
print(args)


# ===================== DATA =====================
df_train = pd.read_csv('data/ddi_training.csv')
df_val   = pd.read_csv('data/ddi_validation.csv')
df_test  = pd.read_csv('data/ddi_test.csv')

train_tup = list(zip(df_train['d1'], df_train['d2'], df_train['type']))
val_tup   = list(zip(df_val['d1'], df_val['d2'], df_val['type']))
test_tup  = list(zip(df_test['d1'], df_test['d2'], df_test['type']))

train_data = DrugDataset(train_tup, ratio=args.data_size_ratio, neg_ent=args.neg_samples)
val_data   = DrugDataset(val_tup, ratio=args.data_size_ratio, disjoint_split=False)
test_data  = DrugDataset(test_tup, disjoint_split=False)

print(f"Training with {len(train_data)} samples")
print(f"Validating with {len(val_data)} samples")
print(f"Testing with {len(test_data)} samples")

train_loader = DrugDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader   = DrugDataLoader(val_data, batch_size=args.batch_size * 3)
test_loader  = DrugDataLoader(test_data, batch_size=args.batch_size * 3)


# ===================== METRICS =====================
def compute_metrics(pred, gt):
    pred_binary = (pred >= 0.5).astype(int)

    acc = metrics.accuracy_score(gt, pred_binary)
    auc = metrics.roc_auc_score(gt, pred)
    auprc = metrics.average_precision_score(gt, pred)
    f1 = metrics.f1_score(gt, pred_binary)

    return acc, auc, auprc, f1


def compute_batch(batch, model, device):
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


# ===================== TRAIN FUNCTION =====================
def train(model, optimizer, loss_fn, scheduler):

    best_val_auc = 0

    print("\nStarting training at:", datetime.now())

    for epoch in range(1, args.n_epochs + 1):
        start = time.time()

        # ================= TRAIN =================
        model.train()
        total_loss = 0

        for batch in train_loader:
            p_score, n_score, _, _ = compute_batch(batch, model, device)

            loss, _, _ = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        # Average training loss
        train_loss = total_loss / len(train_loader)

        # ================= TRAIN METRICS (EVAL MODE) =================
        model.eval()
        train_pred, train_gt = [], []

        with torch.no_grad():
            for batch in train_loader:
                _, _, probas, labels = compute_batch(batch, model, device)
                train_pred.append(probas)
                train_gt.append(labels)

        train_pred = np.concatenate(train_pred)
        train_gt = np.concatenate(train_gt)

        train_acc, train_auc, train_auprc, train_f1 = compute_metrics(train_pred, train_gt)

        # ================= VALIDATION =================
        val_pred, val_gt = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                p_score, n_score, probas, labels = compute_batch(batch, model, device)
                loss, _, _ = loss_fn(p_score, n_score)

                val_loss += loss.item()
                val_pred.append(probas)
                val_gt.append(labels)

        val_loss = val_loss / len(val_loader)

        val_pred = np.concatenate(val_pred)
        val_gt = np.concatenate(val_gt)

        val_acc, val_auc, val_auprc, val_f1 = compute_metrics(val_pred, val_gt)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

        # Step scheduler AFTER optimizer updates
        scheduler.step()

        # ================= PRINT =================
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

    print("\nBest Validation ROC-AUC:", round(best_val_auc, 4))


# ===================== TEST FUNCTION =====================
def test(model):
    print("\n================ TEST RESULTS ================")

    model.eval()
    test_pred, test_gt = [], []

    with torch.no_grad():
        for batch in test_loader:
            _, _, probas, labels = compute_batch(batch, model, device)
            test_pred.append(probas)
            test_gt.append(labels)

    test_pred = np.concatenate(test_pred)
    test_gt = np.concatenate(test_gt)

    acc, auc, auprc, f1 = compute_metrics(test_pred, test_gt)

    print("Test Accuracy :", round(acc, 4))
    print("Test ROC-AUC  :", round(auc, 4))
    print("Test AUPRC    :", round(auprc, 4))
    print("Test F1-score :", round(f1, 4))


# ===================== MAIN =====================
model = models.MASMDDI(
    args.n_atom_feats,
    args.n_atom_hid,
    args.rel_total,
    args.num_layers
).to(device)

loss_fn = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: 0.96 ** e)

train(model, optimizer, loss_fn, scheduler)

# Load best model
model.load_state_dict(torch.load("best_model.pth"))
test(model)
Print("hello world")
