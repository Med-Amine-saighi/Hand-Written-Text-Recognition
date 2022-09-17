import torch
from ctc_decoder import best_path, beam_search
import torch
import Preprocess


def collate(batch):
    images, words = [b.get('images') for b in batch], [b.get('targets') for b in batch]
    images = torch.stack(images, 0)

    lens = [len(item['targets']) for item in batch]
    
    list_of_chars_digitized = [word for word in words]
    lengths = len(list_of_chars_digitized)

    lengths = torch.tensor(lengths, dtype=torch.long)
    lens = torch.tensor(lens, dtype=torch.long)

    for i in range(lengths.item()):
        targets = torch.cat(words)

    return  images,targets, lens

def decode_predictions(preds, encoder):

    preds = preds.permute(1, 0, 2) 
    preds = torch.softmax(preds, 2)

    preds = torch.argmax(preds, 2)

    preds = preds.numpy()

    cap_preds = []
    
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            if k == len(Preprocess.lbl_encoder.classes_):
                temp.append("°")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp)
        cap_preds.append(tp)
    return cap_preds

def decode_predictions_2(preds, encoder):
    preds = torch.softmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    for i in range(preds.shape[1]):
        aux =  preds[:,i,:] 
        print(f'Beam search: "{beam_search(aux,Preprocess.lbl_encoder.classes_)}"')
        break
    
    return None