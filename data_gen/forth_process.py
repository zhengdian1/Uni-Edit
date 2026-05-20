import json
import os
import hashlib 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from nano_api import generate

file_path = 'Custom-Uni-Edit/3.json'
save_file_dir = 'Custom-Uni-Edit/edited'

concurrency = 30 

def get_save_path(image_path, instruction):
    if not image_path:
        return None
    
    if instruction:
        instr_hash = hashlib.md5(instruction.encode('utf-8')).hexdigest()[:8]
    else:
        instr_hash = "no_instr"

    relative_path = "/".join(image_path.split('/')[-2:]) 
    dir_name = os.path.dirname(relative_path)
    base_name = os.path.basename(relative_path)
    file_stem, ext = os.path.splitext(base_name) # file_stem="test", ext=".jpg"
    
    new_filename = f"{file_stem}_{instr_hash}{ext}"
    
    full_save_path = os.path.join(save_file_dir, dir_name, new_filename)
    
    return full_save_path

def process_single_image(message):
    image_path = message.get('image_path')
    instruction = message.get('edit_instruction', '')
    orig_q = message.get('original_question', '')
    proc_a = message.get('process_answer', '')

    save_path = get_save_path(image_path, instruction)
    
    if not save_path:
        return False, f"path parsing failed - {image_path}"

    if os.path.exists(save_path):
        return True, f"file exists"

    prompt_text = (
        f"Context Information:\n"
        f"Original Question: {orig_q}\n"
        f"Reference Answer: {proc_a}\n\n"
        f"Task Instruction:\n"
        f"{instruction}\n\n"
        f"Requirement: Please execute the Task Instruction based on the Reference Answer provided above."
    )
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        generate(
            prompt_text, 
            save_path, 
            [image_path], 
        )
        if os.path.exists(save_path):
            return True, f"success: {save_path}"
        else:
            raise
    except:
        return False, f"failed"

if __name__ == "__main__":
    if not os.path.exists(file_path):
        print(f"no config file: {file_path}")
        exit()

    with open(file_path, 'r', encoding='utf-8') as f:
        all_messages = json.load(f)

    pending_messages = []
    skipped_count = 0
    
    for msg in tqdm(all_messages, desc="pre-checking"):
        img_path = msg.get('image_path')
        instr = msg.get('edit_instruction', '')
        
        save_p = get_save_path(img_path, instr)
        
        if save_p and os.path.exists(save_p):
            skipped_count += 1
        else:
            pending_messages.append(msg)

    print(f"\njump existed: {skipped_count}")
    print(f"need to process: {len(pending_messages)}")
    
    if not pending_messages:
        exit()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_msg = {executor.submit(process_single_image, msg): msg for msg in pending_messages}
        pbar = tqdm(total=len(pending_messages), desc="processing")
        
        success_cnt = 0
        fail_cnt = 0

        for future in as_completed(future_to_msg):
            success, info = future.result()
            if success:
                success_cnt += 1
            else:
                fail_cnt += 1
                tqdm.write(f"❌ {info}")
            
            pbar.update(1)
            pbar.set_postfix({"OK": success_cnt, "Err": fail_cnt})
                
        pbar.close()