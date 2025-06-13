# full_image_analysis_script.py
import os
import random
import base64
import argparse
import itertools
from openai import OpenAI # Uses the modern OpenAI library (v1.0+)
from pathlib import Path
from tqdm import tqdm
# Global variable for API client, initialized in main()
client = None

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

# --- Helper Functions (from minimized script) ---

def get_mime_type(image_filename):
    """Determines the MIME type based on the image file extension."""
    ext = os.path.splitext(os.path.basename(image_filename))[1].lower()
    if ext == '.png':
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    return 'application/octet-stream' # Default

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"ERROR encoding image {image_path}: {e}")
        return None

def call_openai_vision_api(prompt_text, images_data, model_name, max_tokens_api):
    """Calls the OpenAI Vision API using the initialized client."""
    if not client:
        print("ERROR: OpenAI client not initialized.")
        return None

    messages_content = [{"type": "text", "text": prompt_text}]
    for img_data in images_data:
        base64_string = img_data.get("base64")
        mime_type = img_data.get("mime_type")
        if not base64_string or len(base64_string) < 100:
            print("WARNING: Potentially invalid or empty base64 string for an image. Skipping this image in API call.")
            continue # Skip this problematic image data
        if not mime_type:
            print(f"WARNING: MIME type not provided for an image. Defaulting to image/jpeg.")
            mime_type = 'image/jpeg'
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_string}"}
        })
    
    # Check if there are any images to send after validation
    if len(messages_content) == 1 and images_data: # Only text prompt, but images were provided and all failed validation
        print("ERROR: No valid images to send to API after encoding/validation.")
        return None
    if not images_data and len(messages_content) > 0: # No images were intended to be sent, only prompt
        pass # This is fine for text-only prompts, though not used in this script's core logic

    try:
        # print(f"DEBUG: Calling API with model '{model_name}'. Prompt (start): {prompt_text[:200]}...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": messages_content}],
            max_tokens=max_tokens_api
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"ERROR calling OpenAI API with model {model_name}: {e}")
        return None

def extract_object_name_from_filename(image_filename_full):
    """Extracts a simplified object name from the image filename."""
    filename_base = os.path.splitext(os.path.basename(image_filename_full))[0]
    parts = filename_base.split('_')
    if len(parts) > 1:
        for i, part in enumerate(parts):
            if part.startswith("obj") and part[3:].isdigit():
                if i + 1 < len(parts):
                    return "-".join(parts[i+1:]) 
        return parts[-1] 
    return filename_base

# --- Main Task Functions ---

def generate_captions_for_folder(folder_path, num_samples_caption, model_name, max_tokens_api, output_caption_dir):
    """
    Generates and saves captions for a specified number of images in a given folder.
    Captions are saved in a subdirectory within output_caption_dir named after the image folder.
    """
    print(f"\nProcessing folder for captioning: {folder_path}")
    folder_basename = os.path.basename(folder_path)

    # Define the specific output path for this folder's captions
    current_folder_caption_output_path = os.path.join(output_caption_dir, folder_basename)
    try:
        Path(current_folder_caption_output_path).mkdir(parents=True, exist_ok=True)
        print(f"  Caption output directory for this folder: {current_folder_caption_output_path}")
    except Exception as e:
        print(f"ERROR: Could not create caption output directory {current_folder_caption_output_path}: {e}")
        return


    try:
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)) and
                          os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    except FileNotFoundError:
        print(f"ERROR: Folder not found at {folder_path}")
        return
    except Exception as e:
        print(f"ERROR listing files in {folder_path}: {e}")
        return

    if not image_files:
        print(f"  No images found in {folder_path}. Skipping captioning for this folder.")
        return

    num_to_sample = min(num_samples_caption, len(image_files))
    if num_to_sample == 0 and num_samples_caption > 0 :
         print(f"  No images to sample in {folder_path} (num_samples_caption={num_samples_caption}, found={len(image_files)}).")
         return
    elif num_to_sample == 0 and num_samples_caption == 0:
        print(f"  Number of samples for captioning is 0. Skipping captioning for {folder_path}.")
        return

    sampled_images_names = random.sample(image_files, num_to_sample)
    print(f"  Sampling {len(sampled_images_names)} images for captioning: {', '.join(sampled_images_names)}")

    all_captions_for_folder_agg = []

    for image_name in sampled_images_names:
        image_path = os.path.join(folder_path, image_name)
        print(f"    Generating caption for: {image_name}")

        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            print(f"    Skipping {image_name} due to encoding error.")
            continue
        
        mime_type = get_mime_type(image_name)
        image_api_data = [{"base64": image_base64, "mime_type": mime_type}]

        prompt = ("Focus solely on the primary vehicle or person in this image. "
                  "Describe its objective physical attributes such as shape, structure, and key components. "
                  "Avoid any assumptions, interpretations, or imaginative descriptions. "
                  "Present the description as a series of short, factual phrases, for example: "
                  "'A [target object type] with [specific shape/structure], featuring [component A], [component B], and [notable detail C].'")
        
        caption = call_openai_vision_api(prompt, image_api_data, model_name, max_tokens_api)

        if caption:
            # Save individual caption to the designated output subdirectory
            caption_file_name_base = os.path.splitext(image_name)[0]
            individual_caption_file_path = os.path.join(current_folder_caption_output_path, f"{caption_file_name_base}_caption.txt")
            try:
                with open(individual_caption_file_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                print(f"      Saved individual caption to: {individual_caption_file_path}")
                all_captions_for_folder_agg.append(f"--- Caption for {image_name} ---\n{caption}\n")
            except Exception as e:
                print(f"      ERROR saving caption for {image_name}: {e}")
        else:
            print(f"      Failed to generate caption for {image_name}.")

    # Save aggregated captions for the folder to the designated output subdirectory
    if all_captions_for_folder_agg:
        aggregated_captions_file_path = os.path.join(current_folder_caption_output_path, f"{folder_basename}_captions_all.txt") # Changed filename for clarity
        try:
            with open(aggregated_captions_file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_captions_for_folder_agg))
            print(f"  Saved aggregated captions for folder {folder_basename} to: {aggregated_captions_file_path}")
        except Exception as e:
            print(f"  ERROR saving aggregated captions for {folder_basename}: {e}")

def generate_differences_between_folders(folder_A_path, folder_B_path, num_pairs_difference, model_name, max_tokens_api, output_diff_dir):
    """
    Generates and saves difference descriptions for image pairs from two folders.
    """
    folder_A_name = os.path.basename(folder_A_path)
    folder_B_name = os.path.basename(folder_B_path)
    print(f"\nProcessing differences between Folder A ('{folder_A_name}') and Folder B ('{folder_B_name}')")

    try:
        images_in_A = [f for f in os.listdir(folder_A_path) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
        images_in_B = [f for f in os.listdir(folder_B_path) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    except Exception as e:
        print(f"ERROR listing images in {folder_A_name} or {folder_B_name}: {e}")
        return

    if not images_in_A:
        print(f"  No images found in Folder A: {folder_A_name}. Skipping this pair.")
        return
    if not images_in_B:
        print(f"  No images found in Folder B: {folder_B_name}. Skipping this pair.")
        return
    
    if num_pairs_difference == 0:
        print(f"  Number of pairs for difference is 0. Skipping difference generation for {folder_A_name} and {folder_B_name}.")
        return

    all_differences_for_pair_agg = []
    
    # Sample num_pairs_difference pairs of images
    # This simple sampling allows images to be re-picked if num_pairs_difference is large,
    # but ensures each pair consists of one image from A and one from B.
    print(f"  Sampling {num_pairs_difference} image pairs for difference generation.")
    for i in range(num_pairs_difference):
        if not images_in_A or not images_in_B: # Should not happen if initial checks pass
            print("  Cannot sample further, one of the image lists is empty.")
            break 

        image_A_name = random.choice(images_in_A)
        image_B_name = random.choice(images_in_B)
        
        image_A_full_path = os.path.join(folder_A_path, image_A_name)
        image_B_full_path = os.path.join(folder_B_path, image_B_name)

        print(f"    Pair {i+1}: Comparing '{image_A_name}' (from {folder_A_name}) with '{image_B_name}' (from {folder_B_name})")

        image_A_base64 = encode_image_to_base64(image_A_full_path)
        image_B_base64 = encode_image_to_base64(image_B_full_path)

        if not image_A_base64 or not image_B_base64:
            print("      Failed to encode one or both images for this pair. Skipping pair.")
            continue
        
        object_A_name_extracted = extract_object_name_from_filename(image_A_name)
        object_B_name_extracted = extract_object_name_from_filename(image_B_name)
        
        mime_type_A = get_mime_type(image_A_name)
        mime_type_B = get_mime_type(image_B_name)

        images_api_data = [
            {"base64": image_A_base64, "mime_type": mime_type_A},
            {"base64": image_B_base64, "mime_type": mime_type_B}
        ]

        prompt = (
            f"Image 1 is an image of an object referred to as '{object_A_name_extracted}'. "
            f"Image 2 is an image of an object referred to as '{object_B_name_extracted}'.\n\n"
            f"Your task is to compare '{object_A_name_extracted}' (visible in Image 1) and '{object_B_name_extracted}' (visible in Image 2).\n"
            "Focus on their distinct objective physical attributes such as shape, structure, and key components. Ignore color differences.\n\n"
            "Present the differences as a series of short, factual phrases for each object, strictly following this format:\n"
            f"'{object_A_name_extracted}': feature A1, feature A2, feature A3 (or more features if distinct and important).\n"
            f"'{object_B_name_extracted}': feature B1, feature B2, feature B3 (or more features if distinct and important)."
        )
        
        differences_text = call_openai_vision_api(prompt, images_api_data, model_name, max_tokens_api)

        if differences_text:
            image_A_filename_base = os.path.splitext(image_A_name)[0]
            image_B_filename_base = os.path.splitext(image_B_name)[0]

            individual_diff_file_name = f"{folder_A_name}_{image_A_filename_base}-{folder_B_name}_{image_B_filename_base}_diff.txt"
            individual_diff_file_path = os.path.join(output_diff_dir, individual_diff_file_name) # Difference files are in the root of output_diff_dir
            
            try:
                with open(individual_diff_file_path, 'w', encoding='utf-8') as f:
                    f.write(differences_text)
                print(f"      Saved individual differences to: {individual_diff_file_path}")
                
                agg_entry = (f"--- Differences for pair: {image_A_name} (from {folder_A_name}) vs {image_B_name} (from {folder_B_name}) ---\n"
                             f"Object A Name (extracted): {object_A_name_extracted}\n"
                             f"Object B Name (extracted): {object_B_name_extracted}\n"
                             f"{differences_text}\n")
                all_differences_for_pair_agg.append(agg_entry)
            except Exception as e:
                print(f"      ERROR saving individual difference file {individual_diff_file_path}: {e}")
        else:
            print(f"      Failed to generate differences for pair: {image_A_name} and {image_B_name}.")

    # Save aggregated differences for the folder pair
    if all_differences_for_pair_agg:
        aggregated_folder_pair_diff_file_name = f"{folder_A_name}-{folder_B_name}_diff.txt"
        # Difference aggregated files are in the root of output_diff_dir
        aggregated_folder_pair_diff_file_path = os.path.join(output_diff_dir, aggregated_folder_pair_diff_file_name) 
        try:
            with open(aggregated_folder_pair_diff_file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_differences_for_pair_agg))
            print(f"  Saved aggregated differences for folder pair {folder_A_name}-{folder_B_name} to: {aggregated_folder_pair_diff_file_path}")
        except Exception as e:
            print(f"  ERROR saving aggregated differences for {folder_A_name}-{folder_B_name}: {e}")


# --- Main Execution ---
def main():
    global client 

    parser = argparse.ArgumentParser(
        description="Generates image captions and compares images using OpenAI Vision API."
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Parent directory containing the image folders."
    )
    parser.add_argument(
        "--output_caption_dir",
        default="image_captions_output",
        help="Directory to save caption files."
    )
    parser.add_argument(
        "--output_diff_dir", 
        default="image_differences_output", 
        help="Directory to save differentiation files."
    )
    parser.add_argument(
        "--num_samples_caption", 
        type=int, 
        default=100, # Default as per user request
        help="Number of images to sample per folder for captioning."
    )
    parser.add_argument(
        "--num_pairs_difference",
        type=int,
        default=20, # Default as per user request
        help="Number of image pairs to sample per folder-pair for difference generation."
    )
    parser.add_argument(
        "--api_key", 
        default = "sk-a78xK2cAqiMgyAAPD99c229b4114476fB6A46aCa42E98e48", 
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--api_base_url",
        default="https://aihubmix.com/v1",
        help="Optional custom base URL for the OpenAI API (e.g., for a proxy)."
    )
    parser.add_argument(
        "--model_name",
        default="gpt-4.1", 
        help="The model name to use for API calls (e.g., 'gpt-4-turbo')."
    )
    parser.add_argument(
        "--max_tokens_api",
        type=int,
        default=256, 
        help="Maximum number of tokens for API generated responses."
    )

    args = parser.parse_args()

    # Initialize OpenAI client
    api_key_to_use = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key_to_use:
        print("ERROR: OpenAI API key not provided. Set via --api_key argument or OPENAI_API_KEY environment variable.")
        return
    
    try:
        if args.api_base_url:
            print(f"Using custom API base URL: {args.api_base_url}")
            client = OpenAI(api_key=api_key_to_use, base_url=args.api_base_url)
        else:
            print("Using default OpenAI API base URL.")
            client = OpenAI(api_key=api_key_to_use)
        print(f"Using model: {args.model_name}")
        print(f"API Max Tokens: {args.max_tokens_api}")
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' not found.")
        return

    # Create output directories if they don't exist
    try:
        Path(args.output_caption_dir).mkdir(parents=True, exist_ok=True)
        print(f"Caption output main directory: {args.output_caption_dir}")
        Path(args.output_diff_dir).mkdir(parents=True, exist_ok=True)
        print(f"Difference output main directory: {args.output_diff_dir}")
    except Exception as e:
        print(f"ERROR: Could not create output directories: {e}")
        return

    # Get all subdirectories from the data_dir
    try:
        all_subfolders_paths = [
            os.path.join(args.data_dir, d) 
            for d in os.listdir(args.data_dir) 
            if os.path.isdir(os.path.join(args.data_dir, d))
        ]
    except Exception as e:
        print(f"ERROR reading subdirectories from {args.data_dir}: {e}")
        return

    if not all_subfolders_paths:
        print(f"No subfolders found in '{args.data_dir}'. Ensure image folders are direct children of this directory.")
        return
    
    print(f"Found {len(all_subfolders_paths)} image folders: {[os.path.basename(f) for f in all_subfolders_paths]}")

    # --- Task 1: Generate Captions ---
    print("\n--- Starting Task 1: Image Captioning ---")
    for folder_full_path in tqdm(all_subfolders_paths):
        generate_captions_for_folder(
            folder_full_path, 
            args.num_samples_caption, 
            args.model_name, 
            args.max_tokens_api,
            args.output_caption_dir # Pass the new argument
        )

    # --- Task 2: Generate Differences ---
    if len(all_subfolders_paths) >= 2:
        print("\n--- Starting Task 2: Image Differentiation ---")
        # Generate unique pairs of folders
        folder_pairs = list(itertools.combinations(all_subfolders_paths, 2))
        print(f"  Generated {len(folder_pairs)} unique folder pairs for difference comparison.")
        for folder_A_path, folder_B_path in tqdm(folder_pairs):
            generate_differences_between_folders(
                folder_A_path, folder_B_path, 
                args.num_pairs_difference, 
                args.model_name, args.max_tokens_api, 
                args.output_diff_dir
            )
    else:
        print("\nSkipping Task 2: Image Differentiation - Need at least two image folders.")

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()
