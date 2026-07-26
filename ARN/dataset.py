from ctypes import FormatError
import torch
import json
import os
from torch.utils.data import Dataset
from tokenizer_utils import TOKENIZER,TAG_TO_IDX,TOOL_TO_IDX,align_tags_to_tokens
from config import ARNconfig


#mapping the json keys in the dataset with bio tags defined in tag_to_idx
ARG_TO_TAG_BASE={"contact_name":"contact", "message": "message",
                 "text_to_type":"message", "url":"url",
                 "app_name":"app","device":"device",
                 "action": "action", "action_type":"action",
                 "target": "target", "query":"query",
                 "task_query":"query", "minutes":"value",
                 "reminder_message":"reminder", "instruction":"message",
                 "skill_name":"target", "description":"message"}
#function to convert raw user sentence and list of tool calls into word-level bio tags
def generate_word_level_tags(texts:str, tool_calls:list):
    words=texts.split()
    tags=["O"]*len(words) #jitne words utne tags
    word_spans=[]
    current_idx=0
    for word in words:
        start=current_idx
        end=current_idx+len(word)
        word_spans.append((start,end))
        current_idx=end+1
    for tool in tool_calls:
        if not isinstance(tool, dict):
            continue
        arguments=tool.get("arguments",{})
        for arg_name, arg_val in arguments.items():
            arg_val_str=str(arg_val)
            arg_val_cleaned=arg_val_str.strip()
            if not arg_val_cleaned:
                continue
            tag_base_name=ARG_TO_TAG_BASE.get(arg_name)
            if not tag_base_name:
                continue
            start_idx=texts.lower().find(arg_val_cleaned.lower())
            if start_idx==-1:
                continue
            end_idx=start_idx+len(arg_val_cleaned)
            #tracks wherther to use "B-" or "I"
            first_word=True
            for idx, (w_start, w_end) in enumerate(word_spans):
                if max(w_start,start_idx)<min(w_end,end_idx):
                    if first_word:
                        tags[idx]=f"B-{tag_base_name}"
                        first_word=False
                    else:
                        tags[idx]=f"I-{tag_base_name}"
    return words,tags

class ARNDataset(Dataset):
    def __init__(self,jsonl_path:str,tokenizer,max_seq_len:int):
        self.input_ids=[]
        self.attention_masks=[]
        self.tags_ids=[]
        self.tool_labels=[]
        with open(jsonl_path,"r",encoding="utf-8")as f:
            for line in f:
                line=line.rstrip("\n")
                json_obj=json.loads(line)
                messages=json_obj["messages"]
                user_texts=""
                assistant_content=""
                for msg in messages:
                    role=msg.get("role")
                    content=msg.get("content")
                    if role=="user":
                        user_texts=content
                    elif role=="assistant":
                        assistant_content=content
                try:                   
                    tool_calls=json.loads(assistant_content)
                except json.JSONDecodeError:
                    continue
                if not tool_calls:
                    continue
                words,tags=generate_word_level_tags(user_texts,tool_calls)
                input_ids,attention_mask,aligned_tag_ids=align_tags_to_tokens(words=words, tags=tags, tokenizer=TOKENIZER, max_seq_len=max_seq_len)
                tool_vector=[0.0]*ARNconfig.NUM_TOOLS
                for tool in tool_calls:
                    if not isinstance(tool, dict):
                        continue
                    tool_name=tool.get("name")
                    if tool_name in TOOL_TO_IDX:    
                        tool_id=TOOL_TO_IDX[tool_name]
                        tool_vector[tool_id]=1.0
                self.input_ids.append(input_ids)
                self.attention_masks.append(attention_mask)
                self.tags_ids.append(aligned_tag_ids)
                self.tool_labels.append(tool_vector)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "tags_ids": torch.tensor(self.tags_ids[idx], dtype=torch.long),
            "tool_labels": torch.tensor(self.tool_labels[idx], dtype=torch.float)
        }

if __name__ == '__main__':
    # Instantiate your dataset
    dataset = ARNDataset(r"D:\python\AETHER_V1.0\training\aether_orchestrator_dataset.jsonl", TOKENIZER, 32)
    
    # Print the total number of samples loaded
    print(f"Total samples: {len(dataset)}")
    
    # Get the first sample
    sample = dataset[0]
    
    # Print each key in the sample dict and its content or size
    print("Input IDs:", sample["input_ids"])
    print("Tags:", sample["tags_ids"])
    print("Tool Vector:", sample["tool_labels"])





