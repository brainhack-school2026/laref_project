import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

import classifier


def train():

    X_train, X_test, y_train, y_test, feature_cols = classifier.load_prepared_data()  

    # we check which device we are training on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training with {device}")    

    train_loader = classifier.DataLoader(classifier.ABIDEDataset(X_train, y_train), batch_size=classifier.BATCH_SIZE, shuffle=True)
    test_loader  = classifier.DataLoader(classifier.ABIDEDataset(X_test,  y_test),  batch_size=classifier.BATCH_SIZE)

    model = classifier.BrainMLP(input_dim=X_train.shape[1]).to(device)

    # we must apply a normalization step since we have a 60% rate of autism in ou data set
    n_td  = (y_train == 0).sum()
    n_asd = (y_train == 1).sum()
    pos_weight = torch.tensor([n_td / n_asd]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=classifier.LR, weight_decay=classifier.WEIGHT_DECAY)


    
    print("TRAINING : ")
    for epoch in range(classifier.EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss: {total_loss/len(train_loader):.4f}")

    


    torch.save({
        'model_state':  model.state_dict(),
        'input_dim':    X_train.shape[1],
        'feature_cols': feature_cols,
    }, classifier.MODEL_OUT)
    
    
    
    print("Model saved")


    return model
