import math
import torch
from torch import nn

class FactoredAttentionRouter(nn.Module):
    def __init__(self,num_tools:int,hidden_dim:int,dropout:float):
        super().__init__()
        self.num_tools= num_tools
        self.hidden_dim=hidden_dim
        self.tool_embeddings=nn.Embedding(num_embeddings=num_tools,embedding_dim=hidden_dim)
        self.W_Q=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False) 
        self.W_K=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False)
        self.W_V=nn.Linear(in_features=hidden_dim,out_features=hidden_dim,bias=False)
        self.output_proj=nn.Linear(in_features=hidden_dim,out_features=1,bias=False)
        self.dropout= nn.Dropout(dropout)
    def forward(self,encoder_output,attention_mask:None):
        batch_size,seq_len, _ =encoder_output.shape
        tool_indices=torch.arange(0,self.num_tools,device=encoder_output.device)
        tool_indices=self.tool_embeddings(tool_indices)
        Q=self.W_Q(tool_indices)
        K_text=self.W_K(encoder_output)
        V_text=self.W_V(encoder_output)
        Q=Q.unsqueeze(0) #[1,num_tools,hidden_dim]
        #(batch_size,num_tools,seq_len)
        scores=torch.matmul(Q,K_text.transpose(-1,-2))/math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            scores=scores.masked_fill(attention_mask.unsqueeze(1)==0,-1e9)  
        weights=torch.softmax(scores,dim=-1)
        weights=self.dropout(weights)
        #tool context representation
        context=torch.matmul(weights,V_text) #[batch_size,num_tools,hidden_dim]
        context=self.output_proj(context)
        logits=context.squeeze(-1) #[batch_size,num_tools]
        return (logits,weights)

if __name__=='__main__':
    encoder_output=torch.rand((2,20,256),dtype=torch.float32)
    mask=torch.ones((2,20),dtype=torch.bool)
    router=FactoredAttentionRouter(num_tools=16,hidden_dim=256,dropout=0.1)
    logits,weights=router(encoder_output,mask)
    print("shape of logits",logits.shape)
    print("shape of weights",weights.shape)
    print(weights.sum(dim=-1)[0,0])
    params=sum(p.numel() for p in router.parameters())
    print(f"number of parameters: {params:,}")
    
    

