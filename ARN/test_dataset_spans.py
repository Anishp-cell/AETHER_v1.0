import re
from datetime import datetime
from datasets import load_dataset

ds = load_dataset('google/mobile-actions', split='train')

def find_span(text, value):
    if not value or not isinstance(value, str):
        return None
    idx = text.find(value)
    if idx != -1:
        return idx, idx + len(value)
    lower_text = text.lower()
    lower_val = value.lower()
    idx = lower_text.find(lower_val)
    if idx != -1:
        return idx, idx + len(value)
    stripped = value.strip('\'".')
    idx = lower_text.find(stripped.lower())
    if idx != -1:
        return idx, idx + len(stripped)
    return None

# Test datetime span extractor
# Look for temporal expressions in text like:
# "Friday, June 6th, 2025, at 2 PM" or "tomorrow at 3:00 PM" or "next Tuesday at 10 AM"
temporal_pattern = re.compile(
    r'\b(?:on\s+|for\s+|at\s+|this\s+|next\s+|upcoming\s+)*'
    r'(?:(?:this|next|upcoming|coming)\s+)?'
    r'(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|tonight)?'
    r'[\s,]*'
    r'(?:(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}(?:/\d{2,4})?)?'
    r'[\s,]*'
    r'(?:at\s+)?'
    r'(?:\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)|\d{1,2}\s*(?:am|pm)|noon|midnight)?',
    re.IGNORECASE
)

total_args = 0
found_args = 0
total_datetime = 0
found_datetime = 0

for i in range(1000):
    row = ds[i]
    text = row['messages'][1]['content']
    tcs = row['messages'][-1].get('tool_calls') or []
    for tc in tcs:
        for k, v in tc['function']['arguments'].items():
            if v is not None:
                if k == 'datetime':
                    total_datetime += 1
                    # Search for time/date indicators in user text
                    matches = list(temporal_pattern.finditer(text))
                    valid_matches = [m for m in matches if len(m.group(0).strip()) > 3]
                    if valid_matches:
                        found_datetime += 1
                else:
                    total_args += 1
                    span = find_span(text, str(v))
                    if span:
                        found_args += 1

print(f"Non-datetime args: {found_args}/{total_args} ({found_args/total_args*100:.2f}%)")
print(f"Datetime args: {found_datetime}/{total_datetime} ({found_datetime/total_datetime*100:.2f}%)")
