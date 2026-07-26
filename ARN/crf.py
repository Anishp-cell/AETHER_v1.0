from os import path
import torch
import torch
from torch import nn
from tokenizer_utils import TAG_TO_IDX

class CRF(nn.Module):

    def __init__(self,num_tags):
        super().__init__()
        self.num_tags=num_tags
        self.transitions=nn.Parameter(torch.randn((num_tags,num_tags)))
        self.start_transitions=nn.Parameter(torch.randn(num_tags))
        self.end_transitions=nn.Parameter(torch.randn(num_tags))

    def forward_algorithm(self,emissions,mask):
        #emmissions: (batch_size, seq_len, num_tags)
        #mask: (batch_size, seq_len)
        batch_size, seq_len, _ = emissions.shape
        alpha=self.start_transitions.unsqueeze(0)+emissions[:, 0, :]
        for t in range(1,seq_len):
            active=mask[:,t].bool()
            alpha_expanded=alpha.unsqueeze(2)
            trans= self.transitions.unsqueeze(0)
            emit=emissions[:,t,:].unsqueeze(1)
            scores=alpha_expanded+trans+emit #(batch_size,num_tags,num_tags)
            alpha_next=torch.logsumexp(scores,dim=1)
            alpha = torch.where(active.unsqueeze(1), alpha_next, alpha)
        terminal_vars=alpha+self.end_transitions.unsqueeze(0)
        return torch.logsumexp(terminal_vars,dim=1)

    def score_sentence(self,emissions,tags,mask):
        batch_size,seq_len, _ = emissions.shape    
        score=torch.zeros(batch_size,device=emissions.device)
        score=score+self.start_transitions[tags[:,0]]
        score = score + emissions[torch.arange(batch_size), 0, tags[:, 0]] * mask[:, 0]
        for t in range(1,seq_len):
            active = mask[:, t].bool()
            trans_score = self.transitions[tags[:, t-1], tags[:, t]]
            batch_size = emissions.size(0)
            emit_score = emissions[torch.arange(batch_size), t, tags[:, t]]
            step_score = trans_score + emit_score
            score = score + torch.where(active, step_score, torch.zeros_like(score))
        last_active_indices = mask.sum(dim=1).long() - 1
        last_tags = tags[torch.arange(batch_size), last_active_indices]
        score = score + self.end_transitions[last_tags] 
        return score

    def loss(self, emissions,tags,mask):
        forward_score=self.forward_algorithm(emissions,mask)
        gold_score=self.score_sentence(emissions,tags,mask)
        return torch.mean(forward_score-gold_score)

    def viterbi_decode(self,emissions,mask):
        batch_size,seq_len,num_tags=emissions.shape
        viterbi_vars=self.start_transitions.unsqueeze(0)+emissions[:,0,:] #(batch_size,num_tags)
        backpointers=[]
        for t in range(1,seq_len):
            active=mask[:,t].bool()
            next_tag_var=viterbi_vars.unsqueeze(2)+self.transitions.unsqueeze(0)
            max_vars,bptrs=torch.max(next_tag_var,dim=1)
            max_vars=max_vars+emissions[:,t,:]
            viterbi_vars=torch.where(active.unsqueeze(1),max_vars,viterbi_vars)
            backpointers.append(bptrs)
        terminal_vars=viterbi_vars+self.end_transitions.unsqueeze(0)
        best_tag_ids=torch.argmax(terminal_vars,dim=1)
        best_paths=[]
        for b in range(batch_size):
            seq_l=int(mask[b].sum().item())
            best_tag_id=best_tag_ids[b].item()
            path=[best_tag_id]
            for t in range(seq_l - 2, -1, -1):
                bptrs_t = backpointers[t][b] 
                best_tag_id = bptrs_t[best_tag_id].item()
                path.append(best_tag_id)
            path.reverse()
            path.extend([21] * (seq_len - len(path))) 
            best_paths.append(path)
        return best_paths
if __name__=='__main__':
    emissions=torch.randn((2,10,22))
    # mask of shape (2, 10) containing ones and zeros.
    mask=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    tags=torch.randint(0,20,(2,10))
    crf = CRF(num_tags=22)
    loss_val=crf.loss(emissions,tags,mask)
    print(loss_val)
    paths=crf.viterbi_decode(emissions,mask)
    print(paths)
    params=[p.numel() for p in crf.parameters() if p.requires_grad]
    print(sum(params)) 

        
