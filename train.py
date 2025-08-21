

from config import *
from evaluate import evaluate
import torch.nn as nn
import torch
from tqdm import tqdm


def train_model(model, train_dataloader, valid_dataloader, device):
    cum_loss_list = []
    acc_epoch = []
    acc_old = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    # Loop de entrenamiento
    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        cum_loss = 0
        
        
        for i, (label, text, offsets) in enumerate(train_dataloader):
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            cum_loss += loss.item()
        
        cum_loss_list.append(cum_loss)
        accu_val = evaluate(valid_dataloader, model, device)
        acc_epoch.append(accu_val)

        if accu_val > acc_old:
            acc_old = accu_val
            torch.save(model.state_dict(), MODEL_PATH)
        
        
        print(f"Epoch {epoch} | Loss: {cum_loss:.4f} | Val Acc: {accu_val:.4f}")


    return cum_loss_list, acc_epoch

