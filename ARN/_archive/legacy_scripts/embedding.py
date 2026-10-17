import math
import torch
from torch import nn

class FactorizedEmbedding(nn.Module):
    def __init__(self,vocab_size:int,embed_dim_small:int,hidden_dim:int,max_seq_len:int,dropout:float):
        super().__init__() #initializing the parent nn.module class
        self.embedding=nn.Embedding(num_embeddings=vocab_size,
                                    embedding_dim=embed_dim_small)
        self.projection=nn.Linear(in_features=embed_dim_small,
                                    out_features=hidden_dim,
                                    bias=False)
        self.dropout=nn.Dropout(dropout)                   
        self.layer_norm= nn.LayerNorm(hidden_dim)
        position=torch.arange(0,max_seq_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,hidden_dim,2).float() * (-math.log(10000.0)/hidden_dim))
        pe=torch.zeros(max_seq_len,hidden_dim)      
        pe[:,0::2]=torch.sin(position*div_term)#even terms of pe are to the sine of the position
        pe[:,1::2]=torch.cos(position*div_term)#odd terms of pe are to the cosine of the position
        pe=pe.unsqueeze(0)
        #final shape of positional encoding=(1,max_seq_len,hidden_dim)
        self.register_buffer(name='pe',tensor= pe)
    
    def forward(self,input_ids:torch.Tensor):
        seq_len=input_ids.shape[1]
        x=self.embedding(input_ids) #(batch_size,seq_len,emb_dim_small)
        x=self.projection(x) #(batch_size,seq_len,hidden_dim)
        x=x+self.pe[:,:seq_len,:] #adding positional encoding slice wise to x        
        x=self.dropout(x)
        x=self.layer_norm(x)
        return x

if __name__=='__main__':
    input_ids=torch.randint(0,30522,(2,20))
    print(input_ids.shape)
    embed_layer=FactorizedEmbedding(vocab_size=30522,
                                    embed_dim_small=64,
                                    hidden_dim=256,
                                    max_seq_len=32,
                                    dropout=0.1)
    output_embed_layer=embed_layer(input_ids)
    print("output tensor values:\n", output_embed_layer)
    print("output tensor shape:\n", output_embed_layer.shape)
    params=sum(p.numel() for p in embed_layer.parameters())
    print("number of parameters:\n", params)