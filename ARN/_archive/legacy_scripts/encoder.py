from torch._inductor.utils import output_node
import torch
from torch import nn
from attention import MultiHeadSelfAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim:int,num_heads:int, ffn_dim:int, dropout:float):
        super().__init__()
        self.attention= MultiHeadSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.ffn= nn.Sequential(nn.Linear(hidden_dim,ffn_dim), 
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ffn_dim,hidden_dim), 
        nn.Dropout(dropout))
        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
    def forward(self,x,attention_mask:None):
        residual=x
        x=self.norm1(x)
        x=self.attention(x,attention_mask)
        x=x+residual #1st residual connection
        residual=x
        x=self.norm2(x)
        x=self.ffn(x)
        x=x+residual #2nd residual connection
        return x
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers:int, hidden_dim:int, num_heads:int, ffn_dim:int, dropout:float):
        super().__init__()
        self.layers=nn.ModuleList([TransformerEncoderBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(num_layers)])
        self.final_norm=nn.LayerNorm(hidden_dim)
    def forward(self,x,attention_mask=None):
        for layer in self.layers:
            x=layer(x,attention_mask)
        x=self.final_norm(x)
        return x
if __name__ =='__main__':
    x=torch.randn((2,20,256))
    mask=torch.ones((2,20))
    mask[1,15:]=0
    encoder=TransformerEncoder(num_layers=2, hidden_dim=256, num_heads=4, ffn_dim=1024, dropout=0.1)
    output=encoder(x,mask)
    print(output)
    print("Output shape: ", output.shape)
    params=sum(p.numel() for p in encoder.parameters())
    print(f"number of parameters: {params:,}")
    