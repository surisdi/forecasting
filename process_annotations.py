from pathlib import Path
import json
from collections import defaultdict


path_data = Path('/proj/vondrick/didac/code/forecasting/data/long_term_anticipation/annotations')

with open(path_data / 'fho_lta_val.json', 'r') as f:
    annotation = json.load(f)

video_clips = defaultdict(list)
for clip in annotation['clips']:
    video_clips[clip['video_uid']].append(clip)

print('hey')

