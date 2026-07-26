from torch._inductor.utils import output_node
from torch import softmax
import math
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,hidden_dim:int, num_heads:int, dropout:float):
        super().__init__() #initialize parent nn.module class
        assert hidden_dim%num_heads==0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim=hidden_dim
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        self.W_Q=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False) #learnable weight mat for querty
        self.W_K=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False)
        self.W_V=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False)
        self.W_O=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False)
        self.dropout=nn.Dropout(dropout)
    def forward(self, x,attention_mask:None):
        batch_size,seq_len,hidden_dim=x.shape
        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)
        Q=Q.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        Q=Q.transpose(1,2)
        K=K.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        K=K.transpose(1,2)
        V=V.reshape(batch_size,seq_len,self.num_heads,self.head_dim)
        V=V.transpose(1,2)
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
            scores=scores.masked_fill(attention_mask==0, -1e9) #masking the padding tokens and -1e9 forces the softmax to give 0 to padding tokens
        attn_weights=softmax(input=scores,dim=-1) #normalizing the scores
        attn_weights=self.dropout(attn_weights)
        context=torch.matmul(attn_weights,V)
        context=context.transpose(1,2).reshape(batch_size,seq_len,self.hidden_dim)#flattent attention heads back into hidden dimentions
        output=self.W_O(context) #final linear projection
        return output

if __name__=='__main__':
    x= torch.randn((2,20,256))
    mask= torch.ones((2,20))
    mask[1,15:]=0
    attn=MultiHeadSelfAttention(hidden_dim=256,num_heads=4,dropout=0.1)
    out_masked=attn(x,mask)
    print("masked output: ", out_masked)
    print("masked output shape", out_masked.shape)
    params=sum(p.numel() for p in attn.parameters())
    print(f"number of parameters: {params:,}")
    
