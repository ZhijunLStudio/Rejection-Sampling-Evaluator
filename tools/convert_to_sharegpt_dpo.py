import os
import json
import argparse
import glob
import random

def convert_to_dpo_format(input_folder, output_file, system_message="You are a helpful circuit assistant.", 
                          image_folder_prefix="", min_score_threshold=80, prompts_file=None):
    """
    Convert JSON files to DPO format for preference learning.
    
    Parameters:
        input_folder: Folder containing JSON files
        output_file: Path for the output DPO format file
        system_message: System message to include with conversations
        image_folder_prefix: Prefix for image paths
        min_score_threshold: Minimum highest score required to include a sample
        prompts_file: JSON file containing a list of human prompts to randomly select from
    """
    dpo_data = []
    custom_prompts = []
    
    # Load custom prompts if a file is provided
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            custom_prompts = json.load(f)
    
    # Process each JSON file
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if no candidates
            if "all_candidates" not in data or not data["all_candidates"]:
                continue
            
            # Find highest and lowest scoring candidates
            candidates = data["all_candidates"]
            
            # Sort candidates by score in descending order
            sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
            
            highest_score = sorted_candidates[0].get("score", 0)
            
            # Skip if highest score doesn't meet threshold
            if highest_score < min_score_threshold:
                continue
            
            # Get highest and lowest scoring candidates
            chosen_candidate = sorted_candidates[0]
            rejected_candidate = sorted_candidates[-1]
            
            # Get the image path
            image_path = ""
            if "image_info" in data and "filename" in data["image_info"]:
                image_filename = data["image_info"]["filename"]
                # Extract folder structure from the original path if needed
                image_parts = image_filename.split("/")
                image_path = os.path.join(image_folder_prefix, image_parts[-1])
            
            # Determine human prompt
            human_prompt = None
            if custom_prompts:
                human_prompt = random.choice(custom_prompts)
            else:
                human_prompt = chosen_candidate.get("prompt_text", "Analyze this diagram.")
            
            # Create DPO conversation object
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": human_prompt + "<image>"
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen_candidate.get("content", "")
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected_candidate.get("content", "")
                },
                "system": system_message,
                "images": [image_path]
            }
            
            dpo_data.append(conversation)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(dpo_data)} conversations to DPO format. Output file: {output_file}")
    print(f"Skipped {len(json_files) - len(dpo_data)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON files to DPO format')
    parser.add_argument('input_folder', help='Folder containing JSON files')
    parser.add_argument('output_file', help='Path for the output DPO format file')
    parser.add_argument('--system', default='You are a helpful circuit assistant.', help='System message')
    parser.add_argument('--image-prefix', default='', help='Image folder prefix')
    parser.add_argument('--min-score', type=float, default=80, help='Minimum highest score threshold')
    parser.add_argument('--prompts-file', help='JSON file containing human prompts')
    
    args = parser.parse_args()
    
    convert_to_dpo_format(args.input_folder, args.output_file, args.system, 
                         args.image_prefix, args.min_score, args.prompts_file)
