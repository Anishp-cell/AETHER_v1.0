import re
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from datasets import load_dataset

ds = load_dataset('google/mobile-actions', split='train')

def parse_reference_datetime(sys_content):
    match = re.search(r'format:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', sys_content)
    if match:
        return datetime.fromisoformat(match.group(1))
    return None

def extract_datetime_span(user_text):
    """Extract only the temporal expression substring from user utterance."""
    # Common temporal keywords and patterns
    pattern = re.compile(
        r'(?:(?:on|for|at|this|next|upcoming|coming)\s+)?'
        r'(?:(?:this|next|upcoming|coming)\s+)?'
        r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)?'
        r'[\s,]*'
        r'(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)'
        r'[\s,]*'
        r'(?:at\s+)?'
        r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}\s*(?:am|pm)|noon|midnight)?',
        re.IGNORECASE
    )
    match = pattern.search(user_text)
    if match:
        return match.group(0).strip()
    return user_text

def resolve_datetime(user_text, ref_dt):
    if not ref_dt:
        ref_dt = datetime.now()
    
    # Base reference date with zeroed time
    base_ref = ref_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    span = extract_datetime_span(user_text)
    
    # Strip leading prepositions
    clean_span = re.sub(r'^(?:on|for|at)\s+', '', span, flags=re.IGNORECASE).strip()
    
    try:
        dt = date_parser.parse(clean_span, fuzzy=True, default=base_ref)
        # Ensure seconds are zeroed
        dt = dt.replace(second=0, microsecond=0)
        return dt
    except Exception:
        pass
    
    # Fallback to direct fuzzy parse on full text with zeroed base
    try:
        dt = date_parser.parse(user_text, fuzzy=True, default=base_ref)
        return dt.replace(second=0, microsecond=0)
    except Exception:
        return None

# Test on 100 samples
correct = 0
total = 0

for i in range(100):
    row = ds[i]
    msgs = row['messages']
    ref_dt = parse_reference_datetime(msgs[0]['content'])
    tcs = msgs[-1].get('tool_calls') or []
    for tc in tcs:
        gt_dt = tc['function']['arguments'].get('datetime')
        if gt_dt is not None:
            total += 1
            user_text = msgs[1]['content']
            pred_dt = resolve_datetime(user_text, ref_dt)
            if pred_dt and pred_dt.strftime('%Y-%m-%dT%H:%M:%S') == gt_dt.strftime('%Y-%m-%dT%H:%M:%S'):
                correct += 1
            else:
                if total <= 8:
                    print(f"Mismatch [{i}]:")
                    print(f"  User:   {user_text}")
                    print(f"  Ground: {gt_dt}")
                    print(f"  Pred:   {pred_dt}")

print(f"\nExact Match Rate on Datetime Resolution: {correct}/{total} ({correct/total*100:.1f}%)")
