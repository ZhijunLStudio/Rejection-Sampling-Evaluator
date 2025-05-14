import os
import json
import argparse
import glob
import random

def convert_to_sharegpt_format(input_folder, output_file, system_message="You are a helpful circuit assistant.", 
                              image_folder_prefix="images/", min_score=80, prompts_file=None):
    """
    Convert JSON files to ShareGPT format, using the highest-scoring answers.
    
    Parameters:
        input_folder: Folder containing JSON files
        output_file: Path for the output ShareGPT format file
        system_message: System message to use in the conversations
        image_folder_prefix: Prefix for image paths
        min_score: Minimum threshold score for including a conversation
        prompts_file: JSON file containing a list of human prompts to randomly select from
    """
    sharegpt_data = []
    custom_prompts = []
    
    # Load custom prompts if a file is provided
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            custom_prompts = json.load(f)
    
    # Process each JSON file
    json_files = glob.glob(os.path.join(input_folder, "*.json"))
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data contains best_result
            if "best_result" not in data:
                print(f"Skipping file {file_path} - no best_result field")
                continue
            
            best_result = data["best_result"]
            score = best_result.get("score", 0)
            
            # Check if score meets minimum requirement
            if score < min_score:
                print(f"Skipping file {file_path} - score {score} below minimum requirement {min_score}")
                continue
            
            # Determine human prompt
            human_prompt = None
            if custom_prompts:
                human_prompt = random.choice(custom_prompts)
            else:
                human_prompt = best_result.get("prompt_text", "Analyze this circuit diagram.")
            
            # Build image path
            image_path = os.path.join(image_folder_prefix, data["image_info"]["filename"])
            
            # Create ShareGPT conversation object
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": human_prompt + "<image>"
                    },
                    {
                        "from": "gpt",
                        "value": best_result.get("content", "")
                    }
                ],
                "system": system_message,
                "images": [image_path]
            }
            
            sharegpt_data.append(conversation)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(sharegpt_data)} conversations to ShareGPT format. Output file: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON files to ShareGPT format')
    parser.add_argument('input_folder', help='Folder containing JSON files')
    parser.add_argument('output_file', help='Path for the output ShareGPT format file')
    parser.add_argument('--system', default='You are a helpful circuit assistant.', help='System message')
    parser.add_argument('--image-prefix', default='images/', help='Image folder prefix')
    parser.add_argument('--min-score', type=float, default=80, help='Minimum score threshold')
    parser.add_argument('--prompts-file', help='JSON file containing human prompts')
    
    args = parser.parse_args()
    
    convert_to_sharegpt_format(args.input_folder, args.output_file, args.system, 
                            args.image_prefix, args.min_score, args.prompts_file)
