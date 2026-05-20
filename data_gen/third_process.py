# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import json
import argparse
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from system2 import KNOWLEDGE_SYSTEM_PROMPT, ATTR_SYSTEM_PROMPT_YES, ATTR_SYSTEM_PROMPT_SYN, COUNT_SYSTEM_PROMPT_YES, COUNT_SYSTEM_PROMPT_SYN, CAPTION_SYSTEM_PROMPT, LOCATION_SYSTEM_PROMPT, MATH_SYSTEM_PROMPT
import random
import time
import json_repair  # Replaced 're' and manual parsing with json_repair
from openai import RateLimitError, APIError, APITimeoutError

# It is recommended to use environment variables for the API key in official usage
api_key = os.getenv("OPENAI_API_KEY", "your_official_openai_api_key_here")

lock = threading.Lock()
file_path = 'Custom-Uni-Edit/2_filter.json'
OUTPUT_FILE = 'Custom-Uni-Edit/3.json'

def call_gpt(prompts_data, max_retries=5):
    if prompts_data['task_category']=='caption' or prompts_data['task_category']=='ocr':
        UES_SYSTEM_PROMPT = CAPTION_SYSTEM_PROMPT
    elif prompts_data['task_category']=='shape' or prompts_data['task_category']=='color':
        UES_SYSTEM_PROMPT = ATTR_SYSTEM_PROMPT_YES if random.random() < 0.5 else ATTR_SYSTEM_PROMPT_SYN
    elif prompts_data['task_category']=='math':
        UES_SYSTEM_PROMPT = MATH_SYSTEM_PROMPT
    elif prompts_data['task_category']=='knowledge':
        UES_SYSTEM_PROMPT = KNOWLEDGE_SYSTEM_PROMPT
    elif prompts_data['task_category']=='location':
        UES_SYSTEM_PROMPT = LOCATION_SYSTEM_PROMPT
    elif prompts_data['task_category']=='count':
        UES_SYSTEM_PROMPT = COUNT_SYSTEM_PROMPT_YES if random.random() < 0.5 else COUNT_SYSTEM_PROMPT_SYN
    else:
        print(prompts_data['task_category'])
        raise

    if isinstance(prompts_data, dict):
        content_str = json.dumps(prompts_data, ensure_ascii=False)
    else:
        content_str = str(prompts_data)

    for attempt in range(max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[
                    {"role": "system", "content": UES_SYSTEM_PROMPT},
                    {"role": "user", "content": content_str} 
                ]
            )
            return response

        except RateLimitError as e:
            wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"Rate limit hit. Retry {attempt + 1}/{max_retries} after {wait_time:.2f}s...")
            time.sleep(wait_time)

        except (APIError, APITimeoutError) as e:
            wait_time = min(2 ** attempt + random.uniform(0, 1), 30)
            print(f"API error: {e}. Retry {attempt + 1}/{max_retries} after {wait_time:.2f}s...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Non-retryable error: {e}")
            raise

    raise Exception(f"Failed to call GPT after {max_retries} retries.")

def process_single_item(original_item):
    input_for_gpt = {
        "task_category": original_item.get("task_category"),
        "original_question": original_item.get("original_question"),
        "process_answer": original_item.get("process_answer")
    }

    # 2. Call GPT
    try:
        response = call_gpt(input_for_gpt)
        res_content = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    except Exception as e:
        print(f"GPT Call Failed: {e}")
        # Return None to completely skip this item, no error keys added
        return None

    # 3. Parse JSON using json_repair
    try:
        # json_repair.loads automatically handles markdown blocks, missing quotes, trailing commas, etc.
        gpt_result = json_repair.loads(res_content)
        
        if isinstance(gpt_result, dict):
            original_item.update(gpt_result)
        else:
            print(f"JSON Structure Error: Expected dict, got {type(gpt_result)}")
            return None
            
    except Exception as e:
        print(f"JSON Parse Error: {e}")
        return None

    return original_item

def process_json(output_json_path, num_threads, prompts):
    total = len(prompts)
    results = []
    if total == 0:
        print("Nothing to process.")
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit tasks
            future_to_item = {
                executor.submit(process_single_item, item): item
                for item in prompts
            }
            
            for future in tqdm(as_completed(future_to_item), total=total, desc="Processing edits"):
                try:
                    result = future.result()
                    # Skip the item if any error occurred and returned None
                    if result is None:
                        continue
                    if result.get('task_category') == 'yes_or_no':
                        continue
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    import traceback
                    traceback.print_exc()

    # Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"All processing completed. Final result saved in {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--num_processes', type=int, default=40, help="Number of parallel threads")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f) # Note: If it's a standard JSON list, use load; if it's jsonl, use a loop with loads
        
    process_json(OUTPUT_FILE, args.num_processes, prompts)

if __name__ == "__main__":
    model = "gpt-4o-2024-11-20"
    
    # Official OpenAI client initialization
    openai_client = openai.OpenAI(
        api_key=api_key,
    )
    
    main()