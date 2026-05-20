import os
import json
import hashlib 

save_file_dir = 'Custom-Uni-Edit/edited'
file_path = 'Custom-Uni-Edit/3.json'
out_path = 'Custom-Uni-Edit/data.jsonl'

# === 1. Read original data and match generated hash filenames ===
with open(file_path, 'r', encoding='utf-8') as f:
    messages = json.load(f)

final = []
print(f"Total original data: {len(messages)}")

for message in messages:
    image_path = message['image_path']
    instruction = message.get('edit_instruction', '') # Get instruction

    if instruction:
        instr_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
    else:
        instr_hash = "no_instr"

    relative_path = "/".join(image_path.split('/')[-2:]) 
    dir_name = os.path.dirname(relative_path)
    base_name = os.path.basename(relative_path)
    file_stem, ext = os.path.splitext(base_name)
    
    # New filename: test_a1b2c3d4.jpg
    new_filename = f"{file_stem}_{instr_hash}{ext}"
    
    # Concatenate the complete local save path
    save_path = os.path.join(save_file_dir, dir_name, new_filename)
    # --- Core modification end ---

    # Check if the file exists
    if os.path.exists(save_path):
        message['output_image_path'] = save_path
        final.append(message)

print(f"Matched data with existing files: {len(final)}")

# === 2. Check and filter duplicates ===
# Note: Duplicates refer to exact matches of (image + instruction)

seen_suffixes = set()
duplicates = set() # Use set to improve lookup speed

print("Starting duplicate check...")
for item in final:
    full_path = item.get("output_image_path", "")
    if not full_path:
        continue
    
    # Extract the last two levels as a unique identifier (e.g., subdir/test_a1b2c3d4.jpg)
    suffix = "/".join(full_path.split('/')[-2:])
    
    if suffix in seen_suffixes:
        duplicates.add(suffix) # Record duplicate identifier
    else:
        seen_suffixes.add(suffix)

if not duplicates:
    print("No duplicates found.")
else:
    print(f"Found duplicates (to be removed): {len(duplicates)}")


# === 3. Final filtering and saving ===
final_filtered = []
for message in final:
    full_path = message.get("output_image_path", "")
    suffix = "/".join(full_path.split('/')[-2:])
    
    if suffix in duplicates:
        continue
        
    final_filtered.append(message)

print(f"Final valid data count: {len(final_filtered)}")

# Save as JSONL
with open(out_path, 'w', encoding='utf-8') as f:
    for item in final_filtered:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')