import torch
from tqdm import tqdm
import torch
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )

def train_fn(model, data_loader, optimizer):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for xb, yb, lens in tk0:

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        lens = lens.to(DEVICE)
        
        optimizer.zero_grad()
        _, loss = model(xb, lens, yb)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for xb, yb, lens in tk0:
            
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            lens = lens.to(DEVICE)
            
            batch_preds, loss = model(xb, lens, yb)
            fin_loss += loss.item()
            fin_preds.append(batch_preds.detach().cpu())
        return fin_preds, fin_loss / len(data_loader)