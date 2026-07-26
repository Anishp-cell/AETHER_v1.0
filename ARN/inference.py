import re
import json
import torch
from config import ARNconfig
from tokenizer_utils import TOKENIZER, IDX_TO_TAG, IDX_TO_TOOL, TAG_TO_IDX
from model import AetherRoutingNetwork

def predict(text, model, tokenizer, config):
    model.eval()
    
    inputs = tokenizer(
        text,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        tool_preds, tag_preds, routing_weights = model(input_ids, attention_mask)
        
    preds = tool_preds[0]
    active_indices = torch.where(preds)[0].tolist()
    predicted_tools = [IDX_TO_TOOL[idx] for idx in active_indices]
    
    predicted_tag_ids = tag_preds[0]
    subtokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    arguments = {}
    current_tag_base = None
    current_tokens = []
    
    def _save_current_argument():
        if current_tag_base is not None and current_tokens:
            val_str = " ".join(current_tokens)
            clean_val = re.sub(r'\s+##', '', val_str)
            arguments[current_tag_base] = clean_val

    for i in range(len(subtokens)):
        token = subtokens[i]
        tag = IDX_TO_TAG[predicted_tag_ids[i]]
        
        if tag == "<PAD>" or token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if tag.startswith("B-"):
            if current_tag_base is not None:
                _save_current_argument()
            current_tag_base = tag.split("-")[1]
            current_tokens = [token]
            
        elif tag.startswith("I-"):
            tag_type = tag.split("-")[1]
            if current_tag_base == tag_type:
                current_tokens.append(token)
                
        elif tag == "O":
            if current_tag_base is not None:
                _save_current_argument()
            current_tag_base = None
            current_tokens = []
            
    if current_tag_base is not None:
        _save_current_argument()
        
    return {
        "tools": predicted_tools,
        "arguments": arguments
    }

if __name__ == '__main__':
    config = ARNconfig()
    model = AetherRoutingNetwork(config)
    model.load_state_dict(torch.load("checkpoints/arn_best_model.pt", map_location="cpu"))
    model.eval()
    
    test_sentences = [
        "Hey can you just check the phone battery life and then send a whatsapp to my bro saying hi?",
        "Do this: check the laptop battery level, then send a text to my friend that I'm late.",
        "plzzz chk my pc batt level 2 b 4gud & send whats app msg 2 my mom sayin hi",
        "Go to the web and search for who built the Great Pyramid of Giza.",
        "Wait no don't look up the weather, run a shell command instead to clear the log files using rm -rf var log structural entries.",
        "Teach me a new skill called pull metrics where you fetch the local system CPU utilization every 5 seconds using psutil and append it to an analytics file.",
        "Send a quick WhatsApp to my manager... actually make it my director, telling them the deployment went live smoothly.",
        "Search the web for the latest stock price of NVIDIA then run a shell check on our local k8s cluster health status.",
        "Execute a terminal script to clone the git repository at github dot com slash production repo and then pull up a google search for setup errors.",
        "plz teach new skill deploy cluster to run the bash script setup-env sh with full root permissions if needed."
    ]
    
    for sentence in test_sentences:
        prediction = predict(sentence, model, TOKENIZER, config)
        print(f"User Input: {sentence}")
        print(json.dumps(prediction, indent=2))
        print("-" * 60)