import json
import random
from collections import defaultdict

random.seed(42)

input_path = 'Custom-Uni-Edit/2.json'   # Or .jsonl
output_path = 'Custom-Uni-Edit/2_filter.json'

TARGET_LIMITS = {
    "shape": 60000,
    "color": 60000,
    "location": 60000,
    "math": 60000,
    "ocr": 60000,
    "caption": 60000,
    "knowledge": 60000,
    "count": 60000,
    # If there are other categories you don't want to limit, do not include them here, or set them to float('inf')
}

DEFAULT_LIMIT = float('inf') 

def process_data(input_data):
    """
    Input: list of dict
    Output: filtered list of dict
    """
    category_buckets = defaultdict(list)
    for item in input_data:
        cat = item.get("task_category", "unknown")
        category_buckets[cat].append(item)

    final_data = []

    for cat, items in category_buckets.items():
        limit = TARGET_LIMITS.get(cat, DEFAULT_LIMIT)
        count = len(items)
        if count > limit:
            print(f"Category '{cat}' count is {count}, exceeding limit {limit}, performing random sampling...")
            selected_items = random.sample(items, limit)
        else:
            print(f"Category '{cat}' count is {count}, not exceeding limit {limit}, keeping all.")
            selected_items = items
            
        final_data.extend(selected_items)

    # 3. Finally, shuffle the overall order to avoid clustering of the same category data
    random.shuffle(final_data)
    
    return final_data

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_result = process_data(data)
print(f"Processing completed, original count: {len(data)} -> filtered count: {len(filtered_result)}")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_result, f, ensure_ascii=False, indent=4)