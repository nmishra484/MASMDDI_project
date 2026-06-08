import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from sklearn import metrics

import models
from custom_loss import SigmoidLoss

from data_preprocessing import (
    DrugDataset,
    DrugDataLoader,
    TOTAL_ATOM_FEATS
)

# =====================================================
# ARGUMENTS
# =====================================================

parser=argparse.ArgumentParser()

parser.add_argument(
    '--n_epochs',
    type=int,
    default=30
)

parser.add_argument(
    '--setting',
    type=str,
    default='inductive',
    choices=['inductive','transductive']
)

parser.add_argument(
    '--threshold',
    type=float,
    default=0.45
)

parser.add_argument(
    '--lambda_contrastive',
    type=float,
    default=0.1
)

args=parser.parse_args()


# =====================================================
# DATA SPLIT
# =====================================================

def create_split(
        df,
        ratio=0.2,
        setting='inductive'
):

    # ==========================
    # TRANSDUCTIVE
    # ==========================

    if setting=="transductive":

        data=list(
            zip(
                df['d1'],
                df['d2'],
                df['type']
            )
        )

        random.shuffle(data)

        split=int(
            len(data)*(1-ratio)
        )

        train=data[:split]

        test=data[split:]

        print(
            "\nTransductive split"
        )

        print(
            "Train:",
            len(train)
        )

        print(
            "Test:",
            len(test)
        )

        return train,[],test


    # ==========================
    # INDUCTIVE
    # ==========================

    drugs=list(
        set(df['d1']).union(
            set(df['d2'])
        )
    )

    random.shuffle(drugs)

    split=int(
        len(drugs)*(1-ratio)
    )

    old=set(
        drugs[:split]
    )

    new=set(
        drugs[split:]
    )

    train=[]
    s1=[]
    s2=[]

    for _,row in df.iterrows():

        d1=row['d1']
        d2=row['d2']
        r=row['type']

        if d1 in old and d2 in old:

            train.append(
                (d1,d2,r)
            )

        elif d1 in new and d2 in new:

            s1.append(
                (d1,d2,r)
            )

        else:

            s2.append(
                (d1,d2,r)
            )

    print(
        "\nInductive split"
    )

    print(
        "Train:",
        len(train)
    )

    print(
        "S1:",
        len(s1)
    )

    print(
        "S2:",
        len(s2)
    )

    return train,s1,s2


# =====================================================
# TRAIN/VAL SPLIT
# =====================================================

def split_train_val(
        data,
        ratio=0.1
):

    random.shuffle(data)

    split=int(
        len(data)*(1-ratio)
    )

    return (
        data[:split],
        data[split:]
    )


# =====================================================
# METRICS
# =====================================================

def compute_metrics(
        pred,
        gt
):

    pred_bin=(
        pred>=args.threshold
    ).astype(int)

    acc=metrics.accuracy_score(
        gt,
        pred_bin
    )

    auc=metrics.roc_auc_score(
        gt,
        pred
    )

    auprc=metrics.average_precision_score(
        gt,
        pred
    )

    f1=metrics.f1_score(
        gt,
        pred_bin
    )

    return acc,auc,auprc,f1


# =====================================================
# BATCH
# =====================================================

def compute_batch(
        batch,
        model,
        device
):

    pos,neg=batch

    pos=[
        x.to(device)
        for x in pos
    ]

    neg=[
        x.to(device)
        for x in neg
    ]

    p_score,z_h1,z_t1=model(pos)

    n_score,z_h2,z_t2=model(neg)

    prob=torch.sigmoid(
        torch.cat(
            [p_score,n_score]
        )
    ).detach().cpu().numpy()

    gt=np.concatenate(
        [
            np.ones(len(p_score)),
            np.zeros(len(n_score))
        ]
    )

    return (
        p_score,
        n_score,
        prob,
        gt,
        z_h1,
        z_t1
    )


# =====================================================
# TEST
# =====================================================

def test(
        model,
        loader,
        device,
        name
):

    model.eval()

    pred=[]
    gt=[]

    with torch.no_grad():

        for batch in loader:

            _,_,prob,g,_,_=compute_batch(
                batch,
                model,
                device
            )

            pred.append(prob)
            gt.append(g)

    pred=np.concatenate(pred)
    gt=np.concatenate(gt)

    acc,auc,auprc,f1=compute_metrics(
        pred,
        gt
    )

    print(f"\n===== {name} =====")
    print("Accuracy :",round(acc,4))
    print("ROC-AUC  :",round(auc,4))
    print("AUPRC    :",round(auprc,4))
    print("F1-score :",round(f1,4))


# =====================================================
# TRAIN
# =====================================================


# =====================================================
# TRAIN
# =====================================================

def train(
    model,
    train_loader,
    val_loader,
    device
):

    # ==========================================
    # LOSS
    # ==========================================

    criterion=SigmoidLoss()

    # ==========================================
    # OPTIMIZER
    # ==========================================

    optimizer=optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-4
    )

    best_auc=0

    for epoch in range(
        1,
        args.n_epochs+1
    ):

        # =========================
        # TRAINING
        # =========================

        model.train()

        train_loss=0

        train_pred=[]

        train_gt=[]

        for batch in train_loader:

            p_score,n_score,prob,g,z_h,z_t=compute_batch(
                batch,
                model,
                device
            )

            # =====================================
            # CORRECT LOSS CALL
            # =====================================

            total_loss,_,_=criterion(
                p_score,
                n_score
            )

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            train_loss += total_loss.item()

            train_pred.append(prob)

            train_gt.append(g)

        train_pred=np.concatenate(
            train_pred
        )

        train_gt=np.concatenate(
            train_gt
        )

        train_acc,train_auc,train_auprc,train_f1=compute_metrics(
            train_pred,
            train_gt
        )

        # =========================
        # VALIDATION
        # =========================

        model.eval()

        val_loss=0

        val_pred=[]

        val_gt=[]

        with torch.no_grad():

            for batch in val_loader:

                p_score,n_score,prob,g,z_h,z_t=compute_batch(
                    batch,
                    model,
                    device
                )

                # =====================================
                # CORRECT LOSS CALL
                # =====================================

                total_loss,_,_=criterion(
                    p_score,
                    n_score
                )

                val_loss += total_loss.item()

                val_pred.append(prob)

                val_gt.append(g)

        val_pred=np.concatenate(
            val_pred
        )

        val_gt=np.concatenate(
            val_gt
        )

        val_acc,val_auc,val_auprc,val_f1=compute_metrics(
            val_pred,
            val_gt
        )

        # =========================
        # PRINT RESULTS
        # =========================

        print(
            f"\nEpoch {epoch}"
        )

        print(
            f"Train -> "
            f"Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"AUC: {train_auc:.4f} | "
            f"AUPRC: {train_auprc:.4f} | "
            f"F1: {train_f1:.4f}"
        )

        print(
            f"Val   -> "
            f"Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"AUPRC: {val_auprc:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        # =========================
        # SAVE BEST MODEL
        # =========================

        if val_auc > best_auc:

            best_auc=val_auc

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

    print(
        "\nBest Validation AUC:",
        round(best_auc,4)
    )



# =====================================================
# MAIN
# =====================================================

if __name__=="__main__":

    device='cuda' if torch.cuda.is_available() else 'cpu'

    # =========================
    # LOAD DATA
    # =========================

    df=pd.read_csv(
        "data/ddis.csv"
    )

    # =========================
    # RELATION TOTAL
    # =========================

    REL_TOTAL=len(
        set(
            df['type']
        )
    )

    # =========================
    # CREATE SPLIT
    # =========================

    train_data,s1,s2=create_split(
        df,
        setting=args.setting
    )

    # =========================
    # TRAIN / VAL SPLIT
    # =========================

    train_data,val_data=split_train_val(
        train_data
    )

    print(
        "\nTrain data size:",
        len(train_data)
    )

    print(
        "Validation data size:",
        len(val_data)
    )

    print(
        "First few train samples:"
    )

    print(
        train_data[:5]
    )

    # =========================
    # DATASETS
    # =========================

    train_ds=DrugDataset(
        train_data,
        neg_ent=1
    )

    val_ds=DrugDataset(
        val_data,
        neg_ent=1,
        shuffle=False
    )

    # =========================
    # DATALOADERS
    # =========================

    train_loader=DrugDataLoader(
        train_ds,
        batch_size=128,
        shuffle=True
    )

    val_loader=DrugDataLoader(
        val_ds,
        batch_size=128,
        shuffle=False
    )

    # =========================
    # MODEL
    # =========================

    model=models.MASMDDI(
        TOTAL_ATOM_FEATS,
        256,
        REL_TOTAL
    ).to(device)

    # =========================
    # TRAIN
    # =========================

    train(
        model,
        train_loader,
        val_loader,
        device
    )

    # =========================
    # LOAD BEST MODEL
    # =========================

    model.load_state_dict(
        torch.load(
            "best_model.pth"
        )
    )

    # =========================
    # TEST S1
    # =========================

    if s1 is not None and len(s1)>0:

        s1_ds=DrugDataset(
            s1,
            disjoint_split=False
        )

        s1_loader=DrugDataLoader(
            s1_ds,
            batch_size=256
        )

        test(
            model,
            s1_loader,
            device,
            "S1 (NEW-NEW)"
        )

    # =========================
    # TEST S2
    # =========================

    if s2 is not None and len(s2)>0:

        s2_ds=DrugDataset(
            s2,
            disjoint_split=False
        )

        s2_loader=DrugDataLoader(
            s2_ds,
            batch_size=256
        )

        test(
            model,
            s2_loader,
            device,
            "S2 (NEW-OLD)"
        )
        