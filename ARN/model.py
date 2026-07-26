import torch
from torch import nn
from config import ARNconfig
from tokenizer_utils import TAG_TO_IDX
from embedding import FactorizedEmbedding
from encoder import TransformerEncoder
from far import FactoredAttentionRouter
from crf import CRF
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from transformers import AutoModel

class AetherRoutingNetwork(nn.Module):
    def __init__(self,config:ARNconfig):
        super().__init__()
        self.config=config
        # Load pre-trained BERT-tiny as shared trunk
        self.encoder = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
        
        # Freeze the BERT-tiny encoder trunk to prevent catastrophic forgetting
        # Only FAR, CRF, and slot tagger heads will be trained
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.far=FactoredAttentionRouter(num_tools=config.NUM_TOOLS,
                                        hidden_dim=config.HIDDEN_DIM,
                                        dropout=config.DROPOUT)
        self.slot_tagger=nn.Linear(config.HIDDEN_DIM,len(TAG_TO_IDX))
        self.crf=CRF(num_tags=len(TAG_TO_IDX))
    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        # Forward pass through frozen BERT-tiny trunk
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # shape: (B, L, H)
        
        tool_logits,routing_weights=self.far(h,attention_mask)
        emissions=self.slot_tagger(h)
        if tags is not None:
            tool_loss_fn = nn.BCEWithLogitsLoss()
            tool_loss = tool_loss_fn(tool_logits, tool_labels)
            # Per-token normalized slot loss (dividing sequence NLL by active token count)
            active_tokens = attention_mask.sum() + 1e-8
            raw_slot_loss = self.crf.loss(emissions, tags, attention_mask)
            slot_loss = raw_slot_loss / active_tokens
            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_loss, slot_loss, tool_logits, routing_weights)
        else:
            tool_preds=(torch.sigmoid(tool_logits)>0.5)
            tag_preds=self.crf.viterbi_decode(emissions,attention_mask)
            return (tool_preds,tag_preds,routing_weights)
if __name__=='__main__':
    cofnig=ARNconfig()
    model=AetherRoutingNetwork(config=cofnig)
    input_ids=torch.randint(30522,(2,32))
    attention_mask=torch.ones((2,32))
    tool_labels=torch.zeros((2,16))
    tool_labels[0,0]=1.0
    tool_labels[1,1]=1.0
    tags=torch.randint(21,(2,32))
    loss,logits,weights=model(input_ids,attention_mask,tags,tool_labels)
    print("loss is: ", loss)
    model.eval()
    tool_preds,tag_preds,weights=model(input_ids,attention_mask)
    print(tool_preds.shape)
    print(len(tag_preds))
    total_params=sum([p.numel() for p in model.parameters()])
    print("total parameters: ",total_params)       
    
    