import os
import time
import base64
from typing import Any, Dict, List, Optional, Tuple, Union
from google import genai
from google.genai import types
import io
from PIL import Image


API_KEY = ""
BASE_URL = ""
aspect_ratio = "1:1" 
resolution = "1K" 

def generate(
    prompt: Union[Dict[str, Any], str],
    output_image_path: str,
    user_image: Optional[Union[str, List[str]]] = None,
    max_attempts: int = 5,
    retry_delay: float = 2.0
) -> Tuple[List[str], str]:

    client = genai.Client(
        api_key=API_KEY,
        http_options={'base_url': BASE_URL}
    )
    
    output_dir = os.path.dirname(output_image_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    cont = [{"text": prompt}]

    if user_image is not None:
        if isinstance(user_image, str):
            user_image = [user_image]
            
        for item in user_image:
            ext = os.path.splitext(item)[1].lower()
            mime_type = "image/png" if ext == '.png' else "image/jpeg"
            
            with open(item, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            cont.append({"inline_data": {"mime_type": mime_type, "data": image_base64}})

    for attempt in range(max_attempts):
        try:
            if user_image is None:
                response = client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=resolution
                        ),
                    )
                )
            else:
                response = client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=[{"parts": cont}],
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size=resolution
                        ),
                    )
                )
            
            for part in response.parts:
                if image := part.as_image():
                    image.save(output_image_path)
                    return 

        except Exception as e:
            print(f"{attempt + 1} fail: {e}")
            if attempt == max_attempts - 1:
                print("Up to max attempts")
                return
            
            time.sleep(retry_delay)

    return

if __name__ == "__main__":
    prompt_text = 'A cat.'
    user_image_path = None
    target_path = 'nano_pro_demo/result_image.png'
    
    generate(
        prompt=prompt_text,
        output_image_path=target_path,
        user_image=user_image_path
    )
    