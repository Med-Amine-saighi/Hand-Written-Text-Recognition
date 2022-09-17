import torch
import torch
import torch
from  torch import nn
from torch.nn import functional as F
import Preprocess


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear1 = nn.Linear(1600, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images,lens, targets=None):
        bs, c, h, w = images.size()
        #print(bs, c, h, w)
        x = F.relu(self.conv_1(images))
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x) # 1, 64, 18, 75
        x = x.permute(0, 3, 1, 2) # 1, 75 , 64, 18
        x = x.view(bs, x.size(1), -1)
        x = self.linear1(x)
        x = self.drop_1(x)
        #print(x.size()) # torch.Size([1, 75, 64]) -> we have 75 time steps and for each time step we have 64 values
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2) # bs, time steps, values -> CTC LOSS expects it to be

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )

            target_lengths = torch.tensor(
                lens, dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=len(Preprocess.lbl_encoder.classes_))(
                log_probs, targets, input_lengths, lens
            )
            return x, loss

        return x, None