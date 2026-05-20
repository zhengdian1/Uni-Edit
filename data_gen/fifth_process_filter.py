import base64
import os
import json
import argparse
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from system3 import system_prompt1
import random
import time
from io import BytesIO
from typing import Union, Optional, Tuple
from PIL import Image, ImageOps
from openai import RateLimitError, APIError, APITimeoutError
import json_repair  # Added json_repair

# 1. Use Official OpenAI API
api_key = 'your_official_openai_api_key_here' 
model = "gpt-4o-2024-11-20"

openai_client = openai.OpenAI(
    api_key=api_key,
)

lock = threading.Lock()
file_path = 'Custom-Uni-Edit/data.jsonl'
OUTPUT_FILE = 'Custom-Uni-Edit/data_final.jsonl'

def encode_pil_image(pil_image):
    image_stream = BytesIO()
    pil_image.save(image_stream, format='JPEG')
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image

def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    image = Image.open(image)
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    return image

def call_gpt(prompts_data, max_retries=5):
    prompt_content = []
    image = load_image(prompts_data['output_image_path'])
    visual_dict = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_pil_image(image)}"}
        }
    prompt_content.append(visual_dict)

    for attempt in range(max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                stream=False,
                messages=[
                    {"role": "system", "content": system_prompt1},
                    {"role": "user", "content": prompt_content} # Payload content
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
        "output_image_path": original_item.get("output_image_path")
    }

    # 2. Call GPT
    try:
        response = call_gpt(input_for_gpt)
        res_content = response.choices[0].message.content if hasattr(response, "choices") else str(response)
    except Exception as e:
        print(f"GPT Call Failed: {e}")
        # Drop the item if API call fails
        return None

    # 3. Use json_repair to parse and fix the JSON output automatically
    try:
        # json_repair.loads handles markdown blocks and broken JSON structures seamlessly
        gpt_result = json_repair.loads(res_content)
        
        if isinstance(gpt_result, dict):
            # 4. Filter out samples with a score less than 3
            score = float(gpt_result.get("score", 0))
            if score < 3:
                return None
                
            original_item.update(gpt_result)
        else:
            print(f"Parsed result is not a dictionary: {gpt_result}")
            # Drop the item if format is incorrect
            return None
            
    except Exception as e:
        print(f"JSON Parse/Repair Error: {e}")
        # Drop the item if JSON parsing/repairing fails completely
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
                    # Only append valid results (None means it was dropped due to error or low score)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing item: {str(e)}")
                    import traceback
                    traceback.print_exc()

    # Save results
    with open(output_json_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"All processing completed. Final valid items: {len(results)}/{total}. Saved in {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--num_processes', type=int, default=100, help="Number of parallel threads")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    process_json(OUTPUT_FILE, args.num_processes, data)

if __name__ == "__main__":
    main()