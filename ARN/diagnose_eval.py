import torch
from mobile_actions_config import ARNMobileConfig
from mobile_actions_dataset import MobileActionsDataset
from mobile_actions_model import ARNMobileModel
from eval_mobile_actions import (
    parse_reference_datetime, extract_slots_from_tags,
    resolve_datetime_span, TOOL_ARG_SCHEMA, IDX_TO_MOBILE_TOOL
)

config = ARNMobileConfig()
model = ARNMobileModel(config).to('cuda')
model.load_state_dict(torch.load('checkpoints/arn_mobile_best.pt', map_location='cuda'))
model.eval()

eval_dataset = MobileActionsDataset('eval')
tokenizer = eval_dataset.tokenizer

mismatch_reasons = {
    'tool_name_or_order': 0,
    'datetime_diff': 0,
    'token_spacing_or_punct': 0,
    'missing_arg': 0,
    'other_arg_diff': 0
}

samples_shown = 0
for i in range(len(eval_dataset)):
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

    active_indices = torch.where(tool_preds[0])[0].tolist()
    sorted_indices = sorted(active_indices, key=lambda idx: -tool_confs[0, idx].item())
    pred_tools = [IDX_TO_MOBILE_TOOL[idx] for idx in sorted_indices]

    subtokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
    extracted = extract_slots_from_tags(subtokens, tag_preds[0])

    # Check against GT
    gt_tools = [tc['function']['name'] for tc in gt_tcs]
    if pred_tools != gt_tools:
        mismatch_reasons['tool_name_or_order'] += 1
        continue

    # Tools match, check arguments
    all_args_match = True
    for tc in gt_tcs:
        fn = tc['function']['name']
        gt_args = {k: v for k, v in tc['function']['arguments'].items() if v is not None}
        for k, g_val in gt_args.items():
            if k == 'datetime':
                raw_dt = extracted.get('datetime', '')
                p_dt = resolve_datetime_span(raw_dt, ref_dt, user_text)
                g_str = g_val.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(g_val, 'strftime') else str(g_val)
                p_str = p_dt.strftime('%Y-%m-%dT%H:%M:%S') if hasattr(p_dt, 'strftime') else str(p_dt)
                if p_str != g_str:
                    all_args_match = False
                    mismatch_reasons['datetime_diff'] += 1
                    if samples_shown < 4:
                        print(f"=== DATETIME MISMATCH [{i}] ===")
                        print(f"User: {user_text}")
                        print(f"Extracted span: {raw_dt}")
                        print(f"GT datetime:    {g_str}")
                        print(f"Pred datetime:  {p_str}\n")
                        samples_shown += 1
            else:
                p_val = extracted.get(k)
                if p_val is None:
                    all_args_match = False
                    mismatch_reasons['missing_arg'] += 1
                    if samples_shown < 4:
                        print(f"=== MISSING ARG [{i}] ({k}) ===")
                        print(f"User: {user_text}")
                        print(f"GT: {k} = {g_val}")
                        print(f"Extracted: {extracted}\n")
                        samples_shown += 1
                else:
                    p_clean = str(p_val).lower().strip('\'". ')
                    g_clean = str(g_val).lower().strip('\'". ')
                    if p_clean != g_clean:
                        all_args_match = False
                        if p_clean.replace(' ', '') == g_clean.replace(' ', ''):
                            mismatch_reasons['token_spacing_or_punct'] += 1
                        else:
                            mismatch_reasons['other_arg_diff'] += 1
                        if samples_shown < 4:
                            print(f"=== ARG MISMATCH [{i}] ({k}) ===")
                            print(f"User: {user_text}")
                            print(f"GT val:   {g_val}")
                            print(f"Pred val: {p_val}\n")
                            samples_shown += 1

print("Mismatch breakdown:")
for k, v in mismatch_reasons.items():
    print(f"  {k}: {v}")
