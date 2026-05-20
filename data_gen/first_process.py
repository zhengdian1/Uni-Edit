import os
import json
from tqdm import tqdm

input_path = 'llava_onevision_1.5_instruct_22M.jsonl'
output_path = 'Custom-Uni-Edit/1.jsonl'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

valid_count = 0

print("processing...")
with open(input_path, 'r', encoding='utf-8') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:
    
    for line in tqdm(fin):
        line = line.strip()
        if not line: 
            continue
        
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        image_path = obj.get('images', [])
        if len(image_path) != 1:
            continue
        
        messages = obj.get('messages', [])
        if len(messages) < 2: 
            continue
            
        # 3. specific format
        question = messages[0]['content'].split('img_end>\n')[-1]
        answer = messages[1]['content']
        
        json_item = {
            'original_question': question,
            'original_answer': answer,
            'image_path': image_path[0]
        }

        fout.write(json.dumps(json_item, ensure_ascii=False) + '\n')
        valid_count += 1

print(f"Done! Data num: {valid_count}")