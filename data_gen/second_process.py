# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import json
import argparse
import re
import random
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json_repair
import openai
from openai import RateLimitError, APIError, APITimeoutError

from system1 import SYSTEM_PROMPT

lock = threading.Lock()
INPUT_FILE = 'Custom-Uni-Edit/1.jsonl'
OUTPUT_FILE = 'Custom-Uni-Edit/2.json'

def call_gpt(prompts_data, max_retries=10):
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_str} # Must pass a string
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


def process_single_item(prompt):
    # 'prompt' here is the dictionary read from the jsonl file
    response = call_gpt(prompt)
    res = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    
    # Extract JSON string from markdown code blocks if present
    match = re.search(r"```json\s*(.*?)\s*```", res, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = res.strip()
        if json_str.startswith("```"):
            json_str = json_str.strip("`").strip()
            if json_str.startswith("json"):
                json_str = json_str[4:].strip()
    try:
        result = json_repair.loads(json_str)
        return result
    except:
        return res


def process_json(output_json_path, num_threads, prompts):
    total = len(prompts)
    results = []
    
    if total == 0:
        print("Nothing to process.")
        return

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_key = {
            executor.submit(process_single_item, prompt): prompt
            for prompt in prompts
        }
        
        for future in tqdm(as_completed(future_to_key), total=total, desc="Processing edits"):
            try:
                result = future.result()
                
                # Verify if the result is a valid dictionary containing 'task_category'
                if not isinstance(result, dict) or 'task_category' not in result:
                    continue
                
                category = result['task_category']
                # Filter out specific categories
                if category in ['multi-choice', 'others', 'other', 'yes_or_no', 'yes_no']:
                    continue
                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                traceback.print_exc()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"All processing completed. Final result saved in {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--num_processes', type=int, default=30, help="Number of parallel threads")
    args = parser.parse_args()
    
    # Read data from the input JSONL file
    prompts = []
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            prompts = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"Input file {INPUT_FILE} not found.")
        return
        
    process_json(OUTPUT_FILE, args.num_processes, prompts)


if __name__ == "__main__":
    # It is highly recommended to use environment variables for API keys in production
    api_key = os.environ.get("OPENAI_API_KEY", "your_official_openai_api_key_here")
    
    model = "gpt-4o-2024-11-20"
    # model = "gpt-4.1"
    
    openai_client = openai.OpenAI(
        api_key=api_key
    )
    
    main()