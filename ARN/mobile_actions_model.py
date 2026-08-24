import os
import torch
from torch import nn
from transformers import AutoModel

from mobile_actions_config import ARNMobileConfig, NUM_MOBILE_TAGS
from far import FactoredAttentionRouter
from crf import CRF

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

class ARNMobileModel(nn.Module):
    """
    Aether Routing Network configured for Mobile-Actions.
    - Shared frozen BERT-tiny trunk (4.38M params)
    - Factored Attention Router (FAR) for 7 mobile tools
    - Linear-Chain CRF for 20 argument slot BIO tags
    """
    def __init__(self, config: ARNMobileConfig, load_pretrained: bool = True):
        super().__init__()
        self.config = config
        
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(
                "google/bert_uncased_L-2_H-128_A-2",
                attn_implementation="eager"
            )
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = None

        self.far = FactoredAttentionRouter(
            num_tools=config.NUM_TOOLS,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        )
        self.slot_tagger = nn.Linear(config.HIDDEN_DIM, NUM_MOBILE_TAGS)
        self.crf = CRF(num_tags=NUM_MOBILE_TAGS)

    def forward(self, input_ids, attention_mask, tags=None, tool_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state  # shape: (B, L, H)

        tool_logits, routing_weights = self.far(h, attention_mask)
        emissions = self.slot_tagger(h)

        if tags is not None and tool_labels is not None:
            tool_loss_fn = nn.BCEWithLogitsLoss()
            tool_loss = tool_loss_fn(tool_logits, tool_labels)

            active_tokens = attention_mask.sum() + 1e-8
            raw_slot_loss = self.crf.loss(emissions, tags, attention_mask)
            slot_loss = raw_slot_loss / active_tokens

            total_loss = tool_loss + self.config.LAMBDA_SLOT * slot_loss
            return (total_loss, tool_loss, slot_loss, tool_logits, routing_weights)
        else:
            tool_preds = (torch.sigmoid(tool_logits) > 0.5)
            tool_confs = torch.sigmoid(tool_logits)
            tag_preds = self.crf.viterbi_decode(emissions, attention_mask)
            return (tool_preds, tag_preds, tool_confs, routing_weights)
