import torch
from torch import nn

class ShapeError(Exception):
    def __init__(self,ErrorInfo):
        super(ShapeError, self).__init__()
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo

class Transformer_for_SD(nn.Module):
    def __init__(self):
        super(Transformer_for_SD, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024,nhead=2)
        self.transformer = nn.TransformerEncoder(self.encoder_layer,num_layers=6)
        self.pool = nn.AvgPool1d(kernel_size=4,stride=4)
        self.pred_head = nn.Sequential(
            # nn.LayerNorm(4096),
            nn.Linear(4096,4096),
            nn.ReLU(),
            # nn.LayerNorm(4096),
            nn.Linear(4096,2048),
            nn.Sigmoid()
        )

    def forward(self,src):
        if src.shape != torch.Size([16, 64, 1024]):
            raise ShapeError('input tensor should have torch.Size([16, 64, 1024])')

        X = self.transformer(src).permute(1,2,0)
        X = self.pool(X).permute(0,2,1).reshape(64,-1)
        out = self.pred_head(X)

        return out

if __name__=='__main__':
    src = torch.rand(16,64,1024)
    transformer = Transformer_for_SD()
    out = transformer(src)
    print(out.size())
