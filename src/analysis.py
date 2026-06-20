import shap
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, roc_auc_score, precision_recall_curve, 
    auc, balanced_accuracy_score
)

import matplotlib.pyplot as plt

import classifier


def _explain_one_patient(base_value, shap_vals, X_test, feature_cols, sample_idx=0):
    exp = shap.Explanation(
        values       = shap_vals[sample_idx],
        base_values  = base_value,
        data         = X_test[sample_idx],
        feature_names= feature_cols,
    )
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title(f"patient #{sample_idx}")
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_patient{sample_idx}.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 

def explain_model(model_path=classifier.MODEL_OUT):

    # load model
    checkpoint   = torch.load(model_path, map_location="cpu")
    base_model   = classifier.BrainMLP(input_dim=checkpoint["input_dim"])
    base_model.load_state_dict(checkpoint["model_state"])
    feature_cols = checkpoint["feature_cols"]
    wrapped = classifier.ModelWrapper(base_model)
 

    # load data
    X_train, X_test, y_train, y_test, _ = classifier.load_prepared_data()
 
    # setup data
    rng     = np.random.default_rng(42)
    td_idx  = np.where(y_train == 0)[0]
    asd_idx = np.where(y_train == 1)[0]
    n_each  = min(75, len(td_idx), len(asd_idx))
    bg_idx  = np.concatenate([
        rng.choice(td_idx,  n_each, replace=False),
        rng.choice(asd_idx, n_each, replace=False),
    ])
    background = torch.FloatTensor(X_train[bg_idx])
 
    explainer = shap.GradientExplainer(wrapped, background)
    X_test_t  = torch.FloatTensor(X_test)
    shap_raw  = explainer.shap_values(X_test_t)
 
    shap_vals = shap_raw[0] if isinstance(shap_raw, list) else shap_raw
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]
 
    #  we find the average output of the network.
    with torch.no_grad():
        base_value = wrapped(background).mean().item()
 
    # we then rank the features
    mean_abs = np.abs(shap_vals).mean(axis=0)          # (n_features,)
    top_idx  = np.argsort(mean_abs)[::-1]
 


    # Graphing code
    print("Most important features")
    print(f"{'Rank':<5} {'Feature':<45} {'Coefficient':>12}")
    print("-" * 65)
    for rank, i in enumerate(top_idx[:20], 1):
        print(f"  {rank:<4} {feature_cols[i]:<45} {mean_abs[i]:>12.4f}")
 
 
    # Bar plot
    shap.summary_plot(
        shap_vals, X_test,
        feature_names=feature_cols,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.title("Most important features to predict ASD vs TD")
    plt.tight_layout()
    plt.savefig("shap_bar.png", bbox_inches="tight")
    plt.show()


    # we look a 2 patients
    _explain_one_patient(base_value, shap_vals, X_test, feature_cols, sample_idx=10)
    _explain_one_patient(base_value, shap_vals, X_test, feature_cols, sample_idx=23)
 
    return shap_vals, feature_cols, mean_abs





def evaluate_model(model_path=classifier.MODEL_OUT):
    X_train, X_test, y_train, y_test, feature_cols = classifier.load_prepared_data()
    
    checkpoint = torch.load(model_path, map_location="cpu")
    model = classifier.BrainMLP(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_dataset = classifier.ABIDEDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    y_true = []
    y_pred_proba = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred_proba.extend(proba)
            y_true.extend(y_batch.numpy().flatten())

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n" + "="*50)
    print("="*50)
    print(f"Confusion matrix :")
    print(f"          Prédit TD   Prédit ASD")
    print(f"Vrai TD      {tn:>5}       {fp:>5}")
    print(f"Vrai ASD     {fn:>5}       {tp:>5}")


    #  classification
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=['TD', 'ASD']))

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Balanced accuracy : {bal_acc:.3f}")

    # AUC-ROC
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"AUC-ROC : {roc_auc:.3f}")

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"AUC-PR : {pr_auc:.3f}")

    # --- Courbes ROC et PR ---
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.show()

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cm': cm,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'balanced_accuracy': bal_acc
    }


def analyse():

    evaluate_model()
    explain_model()