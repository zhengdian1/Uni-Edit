import os
import glob
import argparse
from safetensors import safe_open
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser(description="Merge HF standard sharded safetensors into a single ema.safetensors")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the downloaded model-*-of-*.safetensors files"
    )
    args = parser.parse_args()

    input_dir = args.model_path
    output_file = os.path.join(input_dir, "ema.safetensors")

    search_pattern = os.path.join(input_dir, "model-*-of-*.safetensors")
    shard_files = sorted(glob.glob(search_pattern))

    if not shard_files:
        print(f"Error: No standard shard files (model-*-of-*.safetensors) found in {input_dir}")
        return

    merged_dict = {}
    print(f"Found {len(shard_files)} shards in '{input_dir}'.")
    print("Starting to merge into ema.safetensors...")

    for file_name in shard_files:
        print(f"Loading {os.path.basename(file_name)}...")
        with safe_open(file_name, framework="pt", device="cpu") as f:
            for key in f.keys():
                merged_dict[key] = f.get_tensor(key)

    print(f"Saving merged weights to {output_file}...")
    print("(This requires ~54GB of free RAM and may take a while)")
    
    save_file(merged_dict, output_file)
    print("Merge completed successfully! You can now run the inference script.")

if __name__ == "__main__":
    main()