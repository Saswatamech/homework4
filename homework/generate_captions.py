import random
from pathlib import Path

import fire
import json
from matplotlib import pyplot as plt

from generate_qa import extract_kart_objects, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, draw_detections, \
    extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    #raise NotImplementedError("Not implemented")

    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        return {
            "image_file": "",
            "candidates": ["Error: Info file not found."],
            "correct_index": 0
        }

    track_name = info_data.get('track', 'Unknown Track')
    kart_objects = extract_kart_objects(info_path, view_index, img_width=ORIGINAL_WIDTH, img_height=ORIGINAL_HEIGHT, min_box_size=10)

    candidates = []
    correct_statement = ""

    # Prioritize generating a correct statement that matches the expected format
    # The expected format examples suggest simple statements about ego car, track, or kart count.

    # Candidate 1: Ego car statement
    center_kart_object = next((k for k in kart_objects if k['is_center_kart']), None)
    if center_kart_object:
        ego_car_name = center_kart_object['kart_name']
        correct_statement_type = random.choice(["ego_car", "num_karts", "track_name"])
    else:
        # If no ego car, we can't make an ego car statement
        correct_statement_type = random.choice(["num_karts", "track_name"])

    if correct_statement_type == "ego_car" and center_kart_object:
        correct_statement = f"{ego_car_name} is the ego car."
    elif correct_statement_type == "num_karts":
        correct_statement = f"There are {len(kart_objects)} karts in the scene."
    elif correct_statement_type == "track_name":
        correct_statement = f"The track is {track_name}."

    candidates.append(correct_statement)

    # Generate incorrect candidates
    all_kart_names = info_data.get('karts', [])
    all_tracks = ["abyss", "fortmagma", "black_forest", "cornfield_crossing", "gran_paradiso_island",
                  "hacienda", "lighthouse", "minigolf", "olivermath", "overworld",
                  "ravenbridge_mansion", "sandtrack", "scotland", "snowmountain",
                  "snowtuxpeak", "tutorial", "volcano_island", "xr591", "zengarden"] # Expanded list of tracks


    # Add incorrect ego car statements
    if center_kart_object:
        incorrect_ego_karts = [k_name for k_name in all_kart_names if k_name != ego_car_name]
        if incorrect_ego_karts:
            candidates.append(f"{random.choice(incorrect_ego_karts)} is the ego car.")
    elif all_kart_names: # If no ego car, still add a random kart name as incorrect
        candidates.append(f"{random.choice(all_kart_names)} is the ego car.")


    # Add incorrect total karts statements
    incorrect_num_karts_options = [str(n) for n in range(max(0, len(kart_objects) - 2), len(kart_objects) + 3) if n != len(kart_objects)]
    if incorrect_num_karts_options:
        candidates.append(f"There are {random.choice(incorrect_num_karts_options)} karts in the scene.")

    # Add incorrect track name statements
    incorrect_tracks_options = [t for t in all_tracks if t != track_name]
    if incorrect_tracks_options:
        candidates.append(f"The track is {random.choice(incorrect_tracks_options)}.")

    # Add incorrect relative position statements (if applicable and a center kart exists)
    other_karts = [k for k in kart_objects if not k['is_center_kart']]
    if center_kart_object and other_karts:
        random_kart_for_incorrect_pos = random.choice(other_karts)
        kart_name_rp = random_kart_for_incorrect_pos['kart_name']

        # Invert position for incorrect candidate
        kart_x_rp = random_kart_for_incorrect_pos['center'][0]
        kart_y_rp = random_kart_for_incorrect_pos['center'][1]
        ego_car_x_rp = center_kart_object['center'][0]
        ego_car_y_rp = center_kart_object['center'][1]

        # Invert x_pos
        inverted_x_pos = 'right of' if kart_x_rp < ego_car_x_rp else 'left of'
        # Invert y_pos
        inverted_y_pos = 'behind' if kart_y_rp < ego_car_y_rp else 'in front of'

        candidates.append(f"{kart_name_rp} is {inverted_y_pos} and {inverted_x_pos} the ego car.")


    # Ensure we have at least 5 unique candidates, add more generic ones if needed
    while len(candidates) < 5:
        generic_incorrect_options = [
            "This is a night scene.",
            "The karts are racing on a street.",
            "There are no karts visible.",
            "The track is a desert."
        ]
        candidates.append(random.choice(generic_incorrect_options))
        candidates = list(set(candidates)) # Remove duplicates

    # Trim to exactly 5 candidates if more than 5 after adding generic ones
    if len(candidates) > 5:
        candidates = random.sample(candidates, 5)
        # Ensure the correct statement is still in candidates after trimming
        if correct_statement not in candidates:
            candidates[random.randint(0, 4)] = correct_statement # Replace a random one with the correct one

    # Shuffle candidates again to randomize correct_index
    random.shuffle(candidates)
    correct_index = candidates.index(correct_statement)

    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_file_path = f"{info_path_obj.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    return {
        'image_file': image_file_path,
        'candidates': candidates,
        'correct_index': correct_index
    }


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
# def generate_caption_file(data_folder: str):
#     """
#     Generates (image_file, caption) pairs for all info files in a given folder
#     and saves them to a single JSON file.
#
#     Args:
#         data_folder (str): Path to the folder containing info.json files (e.g., 'data/train').
#         output_file (str): The name of the output JSON file (e.g., 'generated_captions.json').
#     """
#     folder_path = Path(data_folder)
#     info_files = list(folder_path.glob("*_info.json"))
#
#     if not info_files:
#         print(f"No info.json files found in '{data_folder}'.")
#         return
#
#     for info_file in info_files:
#         print(f"Processing {info_file}...")
#         try:
#             with open(info_file, 'r') as f:
#                 info_data = json.load(f)
#
#             output_file_name = f"{info_file.stem.replace('_info', '')}_captions.json"
#             output_path = folder_path / output_file_name
#
#             if "detections" not in info_data:
#                 print(f"Warning: 'detections' key not found in {info_file}. Skipping.")
#                 continue
#
#             current_info_captions_data = []
#             for view_index in range(len(info_data["detections"])):
#                 caption_entry = generate_caption(str(info_file), view_index)
#                 current_info_captions_data.append(caption_entry)
#
#             with open(output_path, 'w') as f:
#                 json.dump(current_info_captions_data, f, indent=2)
#
#             print(f"Successfully generated {len(current_info_captions_data)} caption entries and saved to {output_path}")
#
#         except Exception as e:
#             print(f"Error processing {info_file}: {e}")

def _generate_single_caption_entry(info_path: str, view_index: int) -> dict:
    """
    Helper function to generate a single caption entry (image_file, caption)
    for a given info file and view index, matching the example_captions.json format.
    """
    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        # Return a placeholder or raise an error as appropriate for a helper
        return {"image_file": "", "caption": "Error: Info file not found."}

    track_name = info_data.get('track', 'Unknown Track')
    kart_objects = extract_kart_objects(info_path, view_index, img_width=ORIGINAL_WIDTH, img_height=ORIGINAL_HEIGHT, min_box_size=10)

    correct_statement = ""

    # Randomly choose which type of caption to generate for this entry
    caption_type_choices = []
    if next((k for k in kart_objects if k['is_center_kart']), None):
        caption_type_choices.append("ego_car")
    if kart_objects: # Only if there are karts to count
        caption_type_choices.append("num_karts")
    caption_type_choices.append("track_name")

    if not caption_type_choices: # Fallback if no specific captions can be generated
        correct_statement = "A scene from the game."
    else:
        chosen_type = random.choice(caption_type_choices)

        if chosen_type == "ego_car":
            center_kart_object = next((k for k in kart_objects if k['is_center_kart']), None)
            if center_kart_object:
                ego_car_name = center_kart_object['kart_name']
                correct_statement = f"{ego_car_name} is the ego car."
        elif chosen_type == "num_karts":
            correct_statement = f"There are {len(kart_objects)} karts in the scene."
        elif chosen_type == "track_name":
            correct_statement = f"The track is {track_name}."

    info_path_obj = Path(info_path)
    # Extract the base name (e.g., '00000') from '00000_info.json'
    base_name = info_path_obj.stem.replace("_info", "")
    # Construct the image file path relative to the data folder (e.g., 'train/00000_00_im.jpg')
    # The parent.name will give 'train' if info_path is 'data/train/00000_info.json'
    image_file_path = f"{info_path_obj.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    return {
        "image_file": image_file_path,
        "caption": correct_statement
    }

def generate_train_caption_data(data_folder: str):
    """
    Generates training caption data (image_file, caption) for all info files
    in a given folder and saves them to a single JSON file,
    matching the format of example_captions.json.

    Args:
        data_folder (str): Path to the folder containing info.json files (e.g., 'data/train').
        output_file (str): The name of the output JSON file (e.g., 'generated_train_captions.json').
    """
    folder_path = Path(data_folder)
    info_files = list(folder_path.glob("*_info.json"))

    if not info_files:
        print(f"No info.json files found in '{data_folder}'.")
        return

    for info_file in info_files:
        print(f"Processing {info_file} for training captions...")
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            # Define the output file name for this specific info_file
            output_file_name = f"{info_file.stem.replace('_info', '')}_captions.json"
            output_path = folder_path / output_file_name

            if "detections" not in info_data:
                print(f"Warning: 'detections' key not found in {info_file}. Skipping.")
                continue

            current_info_train_captions_data = []
            for view_index in range(len(info_data["detections"])):
                # Use the helper function for single caption entries (training format)
                caption_entry = _generate_single_caption_entry(str(info_file), view_index)
                current_info_train_captions_data.append(caption_entry)

            with open(output_path, 'w') as f:
                json.dump(current_info_train_captions_data, f, indent=2)

            print(f"Successfully generated {len(current_info_train_captions_data)} training caption entries and saved to {output_path}")

        except Exception as e:
            print(f"Error processing {info_file}: {e}")

def generate_all_mc_qas_file(data_folder: str, output_file: str = "all_mc_qas.json"):
    """
    Generates (image_file, candidates, correct_index) pairs for all info files in a given folder
    and saves them to a single JSON file (e.g., all_mc_qas.json).
    This function is for generating QA-style data for validation/testing.

    Args:
        data_folder (str): Path to the folder containing info.json files (e.g., 'data/valid').
        output_file (str): The name of the output JSON file (e.g., 'all_mc_qas.json').
    """
    folder_path = Path(data_folder)
    info_files = list(folder_path.glob("*_info.json"))

    if not info_files:
        print(f"No info.json files found in '{data_folder}'.")
        return

    all_qa_data = []
    for info_file in info_files:
        print(f"Processing {info_file} for QA captions...")
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            if "detections" not in info_data:
                print(f"Warning: 'detections' key not found in {info_file}. Skipping.")
                continue

            for view_index in range(len(info_data["detections"])):
                # Use the original generate_caption for QA format
                qa_entry = generate_caption(str(info_file), view_index)
                all_qa_data.append(qa_entry)

        except Exception as e:
            print(f"Error processing {info_file}: {e}")

    output_path = folder_path / output_file
    with open(output_path, 'w') as f:
        json.dump(all_qa_data, f, indent=2)

    print(f"Successfully generated {len(all_qa_data)} QA entries and saved to {output_path}")

def count_training_captions(data_folder: str) -> int:
    """
    Counts the total number of caption pairs across all *_captions.json files
    in a given training data folder.

    Args:
        data_folder (str): The path to the training data folder (e.g., 'data/train').

    Returns:
        The total number of caption pairs, or 0 if no files are found or an error occurs.
    """
    folder_path = Path(data_folder)
    caption_files = list(folder_path.glob("*_captions.json"))

    if not caption_files:
        print(f"No caption files found in '{data_folder}'.")
        return 0

    total_count = 0
    for file_path in caption_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    total_count += len(data)
                else:
                    print(f"Warning: The file '{file_path}' does not contain a list. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from '{file_path}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{file_path}': {e}")

    return total_count

def main():
    fire.Fire({"check": check_caption,"generate_caption":generate_train_caption_data,"generate_validation":generate_all_mc_qas_file,"count_training_captions":count_training_captions})


if __name__ == "__main__":
    main()
