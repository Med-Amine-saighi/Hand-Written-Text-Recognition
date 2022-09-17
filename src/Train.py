import torch
import numpy as np
import pandas as pd
import torch
import numpy as np
from sklearn import model_selection
import torch
from pprint import  pprint
from CFG import BATCH_SIZE ,EPOCHS
from Dataset import  ClassificationDataset,test_transform, train_transform
from Model import CaptchaModel 
from Engine import train_fn ,eval_fn
from Helpers import decode_predictions_2, collate ,decode_predictions
from Preprocess import lbl_encoder, targets_orig


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
train_csv = pd.read_csv('train_csv')

def main():
    (
        train,
        test,
        _,
        test_targets_orig,
    ) = model_selection.train_test_split(
        train_csv, targets_orig, test_size=0.1, random_state=42
    )

    train = train.sample(20).reset_index(drop=True)
    test = test.sample(20).reset_index(drop=True)
    
    train_dataset = ClassificationDataset(
        dataframe=train,
        transform=train_transform
        )
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate
        )
    test_dataset = ClassificationDataset(
        dataframe=test,
        transform=test_transform
        
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate
    )

    model = CaptchaModel(num_chars=len(lbl_encoder.classes_))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    best_score = 0.0 
    best_loss = np.inf

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = eval_fn(model, test_loader)
        valid_cap_preds = []
        for vp in valid_preds:
            decode_predictions_2(vp ,lbl_encoder )
            break
            
        for vp in valid_preds:
            current_preds = decode_predictions(vp, lbl_encoder)
            valid_cap_preds.extend(current_preds)

        pprint(list(zip(test_targets_orig, valid_cap_preds))[0:6])
        print(f"EPOCH: {epoch}   ,    train_loss={train_loss},    valid_loss={valid_loss}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            print("=> saving checkpoint")
            torch.save(model.state_dict(),'Words_best_score.pt')

if __name__ == "__main__":
    main()