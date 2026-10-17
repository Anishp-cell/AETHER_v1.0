import re
import torch
from mobile_actions_config import ARNMobileConfig, IDX_TO_MOBILE_TOOL, IDX_TO_MOBILE_TAG
from mobile_actions_dataset import MobileActionsDataset
from mobile_actions_model import ARNMobileModel
from eval_mobile_actions import parse_reference_datetime, resolve_datetime_span, TOOL_ARG_SCHEMA

config = ARNMobileConfig()
model = ARNMobileModel(config).to('cuda')
model.load_state_dict(torch.load('checkpoints/arn_mobile_best.pt', map_location='cuda'))
model.eval()

eval_dataset = MobileActionsDataset('eval')
tokenizer = eval_dataset.tokenizer

def extract_slots_clean(user_text, input_ids_tensor, tag_preds):
    """
    Extracts slots by finding character spans directly from the raw user text
    rather than joining subtokens with spaces.
    """
    # Use tokenizer's fast token-to-character mapping (offsets)
    tokenized = tokenizer(user_text, return_offsets_mapping=True, max_length=64, truncation=True, padding="max_length")
    offsets = tokenized["offset_mapping"]
    
    slots = {}
    current_tag = None
    span_start = None
    span_end = None
    
    for i, tag_id in enumerate(tag_preds):
        tag = IDX_TO_MOBILE_TAG.get(tag_id, "O")
        start_char, end_char = offsets[i]
        
        # Skip special tokens or padding
        if start_char == end_char or tag == "<PAD>":
            continue
            
        if tag.startswith("B-"):
            if current_tag is not None and span_start is not None:
                val = user_text[span_start:span_end].strip('\'",. ')
                slots[current_tag] = val
            current_tag = tag.split("-")[1]
            span_start = start_char
            span_end = end_char
        elif tag.startswith("I-"):
            tag_type = tag.split("-")[1]
            if current_tag == tag_type:
                span_end = end_char
        elif tag == "O":
            if current_tag is not None and span_start is not None:
                val = user_text[span_start:span_end].strip('\'",. ')
                slots[current_tag] = val
            current_tag = None
            span_start = None
            span_end = None
            
    if current_tag is not None and span_start is not None:
        val = user_text[span_start:span_end].strip('\'",. ')
        slots[current_tag] = val
        
    return slots

strict_matches = 0
name_matches = 0
total = len(eval_dataset)

for i in range(total):
    item = eval_dataset[i]
    raw_row = eval_dataset.ds[i]
    msgs = raw_row['messages']
    user_text = msgs[1]['content']
    ref_dt = parse_reference_datetime(msgs[0]['content'])
    gt_tcs = msgs[-1].get('tool_calls') or []

    input_ids = item['input_ids'].unsqueeze(0).to('cuda')
    attention_mask = item['attention_mask'].unsqueeze(0).to('cuda')

    with torch.no_grad():
        tool_preds, tag_preds, tool_confs, _ = model(input_ids, attention_mask)

    # 1. Active tool indices
    active_indices = torch.where(tool_preds[0])[0].tolist()
    
    # 2. Extract clean slots directly from character offsets
    extracted = extract_slots_clean(user_text, input_ids[0], tag_preds[0])

    # 3. Order tools by text position rather than pure sigmoid confidence
    # Tool mentions in text (find where tool-related keywords or slot values appear)
    def tool_text_pos(idx):
        tool_name = IDX_TO_MOBILE_TOOL[idx]
        keywords = {
            "turn_on_flashlight": ["turn on", "flashlight on", "on the flashlight", "torch on"],
            "turn_off_flashlight": ["turn off", "flashlight off", "off the flashlight", "torch off"],
            "open_wifi_settings": ["wifi", "wi-fi", "network"],
            "create_calendar_event": ["reminder", "calendar", "event", "meeting", "schedule", "appointment"],
            "send_email": ["email", "mail", "send"],
            "create_contact": ["contact", "save", "add"],
            "show_map": ["map", "location", "directions", "where is"]
        }
        for kw in keywords.get(tool_name, []):
            pos = user_text.lower().find(kw)
            if pos != -1:
                return pos
        return 9999

    sorted_indices = sorted(active_indices, key=lambda idx: tool_text_pos(idx))
    pred_tools = [IDX_TO_MOBILE_TOOL[idx] for idx in sorted_indices]

    # Ground truth
    gt_calls = []
    for tc in gt_tcs:
        fn = tc['function']['name']
        args = {k: v for k, v in tc['function']['arguments'].items() if v is not None}
        gt_calls.append({"name": fn, "arguments": args})

    gt_names = [c["name"] for c in gt_calls]
    if pred_tools == gt_names:
        name_matches += 1

    # Check strict exact match
    if len(pred_tools) == len(gt_calls):
        call_match = True
        for p_name, g_call in zip(pred_tools, gt_calls):
            if p_name != g_call["name"]:
                call_match = False
                break
            
            expected_args = TOOL_ARG_SCHEMA.get(p_name, [])
            for k, g_val in g_call["arguments"].items():
                if k == "datetime":
                    raw_dt = extracted.get("datetime", "")
                    p_dt = resolve_datetime_span(raw_dt, ref_dt, user_text)
                    g_str = g_val.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(g_val, 'strftime') else str(g_val)
                    p_str = p_dt.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(p_dt, 'strftime') else str(p_dt)
                    if p_str != g_str:
                        call_match = False
                        break
                else:
                    p_val = extracted.get(k)
                    if p_val is None:
                        call_match = False
                        break
                    p_clean = str(p_val).lower().strip('\'",. ')
                    g_clean = str(g_val).lower().strip('\'",. ')
                    if p_clean != g_clean:
                        call_match = False
                        break

        if call_match:
            strict_matches += 1

print(f"Name-Only Accuracy:           {name_matches}/{total} ({name_matches/total*100:.2f}%)")
print(f"Ordered Strict Exact Match:   {strict_matches}/{total} ({strict_matches/total*100:.2f}%)")
