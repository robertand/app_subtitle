import os
import re

def filter_hallucinations(text):
    if not text:
        return ""
    hallucinations = [
        r"Vă mulțumim pentru vizionare!",
        r"Subtitrare realizată de",
        r"Vă mulțumesc pentru vizionare!"
    ]
    filtered_text = text
    for pattern in hallucinations:
        filtered_text = re.sub(pattern, "", filtered_text, flags=re.IGNORECASE)
    return filtered_text.strip()

def deduplicate_segments(segments):
    if not segments:
        return []
    unique_segments = []
    for i, seg in enumerate(segments):
        if i == 0:
            unique_segments.append(seg)
            continue
        current_text = seg['text'].strip().lower()
        last_text = unique_segments[-1]['text'].strip().lower()
        if current_text == last_text and (seg['start'] - unique_segments[-1]['end']) < 2.0:
            unique_segments[-1]['end'] = seg['end']
        else:
            unique_segments.append(seg)
    return unique_segments

def merge_logic(all_raw_segments):
    final_segments = []
    if all_raw_segments:
        all_raw_segments.sort(key=lambda x: x['start'])
        for seg in all_raw_segments:
            is_duplicate = False
            for existing in final_segments:
                overlap_start = max(seg['start'], existing['start'])
                overlap_end = min(seg['end'], existing['end'])
                overlap_dur = max(0, overlap_end - overlap_start)
                seg_dur = seg['end'] - seg['start']
                existing_dur = existing['end'] - existing['start']
                min_dur = min(seg_dur, existing_dur)
                if min_dur > 0 and (overlap_dur / min_dur) > 0.6:
                    is_duplicate = True
                    if seg.get('avg_logprob', -1e9) > existing.get('avg_logprob', -1e9):
                        existing['text'] = seg['text']
                        existing['avg_logprob'] = seg.get('avg_logprob')
                    break
            if not is_duplicate:
                final_segments.append(seg)
    return final_segments

# Test Data
raw = [
    {'start': 0, 'end': 2, 'text': 'Salut', 'avg_logprob': -0.5},
    {'start': 0.1, 'end': 2.1, 'text': 'Salut!', 'avg_logprob': -0.1}, # Better variant
    {'start': 2, 'end': 4, 'text': 'lume', 'avg_logprob': -0.3},
    {'start': 2.1, 'end': 4.1, 'text': 'lume', 'avg_logprob': -0.5}, # Worse variant
    {'start': 5, 'end': 6, 'text': 'Vă mulțumim pentru vizionare!', 'avg_logprob': -0.1} # Hallucination
]

# Apply filters
for s in raw:
    s['text'] = filter_hallucinations(s['text'])

raw = [s for s in raw if s['text']]

merged = merge_logic(raw)
deduped = deduplicate_segments(merged)

print(f"Final segments: {deduped}")
assert len(deduped) == 2
assert deduped[0]['text'] == 'Salut!'
assert deduped[1]['text'] == 'lume'
print("Test passed!")
