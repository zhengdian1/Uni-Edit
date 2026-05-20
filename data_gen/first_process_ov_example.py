
import os
import json
import random
from tqdm import tqdm

random.seed(42)
# can be any vqa data
input_path = 'llava_onevision_1.5_instruct_22M.jsonl'
output_path = 'Custom-Uni-Edit/first_process_large.jsonl'

target_subjects = [
    "allava_instruct_laion4v", "arxiv_figs", "CLEVR", "CLEVR-Math", 
    "coco", "code_feedback_66k", "geomverse", "ocr", "sherlock", "vg", "vsr"
]

limits = {subj: 20000 for subj in target_subjects} 
limits['CLEVR'] = 120000
limits['CLEVR-Math'] = 120000
limits['vg'] = 120000
limits['vsr'] = 120000
limits['ocr'] = 60000

reservoirs = {subj: [] for subj in target_subjects}
counts = {subj: 0 for subj in target_subjects}

with open(input_path, 'r', encoding='utf-8') as f:
    for i, line in tqdm(enumerate(f)):
        line = line.strip()
        if not line: continue
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        subject = obj.get('subject')

        if subject not in target_subjects:
            continue
            
        image_path = obj.get('images', [])
        if len(image_path) != 1:
            continue
        
        image_path = image_path[0]
        messages = obj.get('messages', [])
        if len(messages) < 2: 
            continue
            
        question = messages[0]['content'].split('img_end>\n')[-1]
        answer = messages[1]['content']
        
        json_item = {
            'original_question': question,
            'original_answer': answer,
            'image_path': image_path,
            'subject': subject 
        }

        limit = limits[subject]
        current_count = counts[subject]
        
        if current_count < limit:
            reservoirs[subject].append(json_item)
        
        else:
            r = random.randint(0, current_count)
            if r < limit:
                reservoirs[subject][r] = json_item
        
        counts[subject] += 1

final_json = []
for subj in reservoirs:
    print(f"Subject: {subj}, Full: {counts[subj]}, final: {len(reservoirs[subj])}")
    final_json.extend(reservoirs[subj])

random.shuffle(final_json)

print(f"Data Num{len(final_json)}...")
with open(output_path, 'w', encoding='utf-8') as f:
    for item in final_json:
        item.pop('subject', None) 
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
