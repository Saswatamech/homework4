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
def _generate_single_caption_entry(info_path: str, view_index: int) -> dict:
    """
    Helper function to generate a single caption entry (image_file, caption)
    for a given info file and view index, matching the example_captions.json format.
    """
    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        return {"image_file": "", "caption": "Error: Info file not found."}

    track_name = info_data.get('track', 'Unknown Track')
    kart_objects = extract_kart_objects(info_path, view_index, img_width=ORIGINAL_WIDTH, img_height=ORIGINAL_HEIGHT, min_box_size=10)

    correct_statement = ""
    # Ensure a caption is always generated
    while correct_statement == "":
        caption_type_choices = []
        center_kart_object = next((k for k in kart_objects if k['is_center_kart']), None)
        other_karts = [k for k in kart_objects if not k['is_center_kart']]

        if center_kart_object:
            caption_type_choices.append("ego_car")
        if kart_objects:
            caption_type_choices.append("num_karts")
        caption_type_choices.append("track_name")
        if center_kart_object and other_karts:
            caption_type_choices.extend(["front_of_ego", "behind_ego", "right_of_ego", "left_of_ego"])

        if not caption_type_choices:
            correct_statement = "A scene from the game."
        else:
            chosen_type = random.choice(caption_type_choices)

            if chosen_type == "ego_car":
                if center_kart_object:
                    correct_statement = f"{center_kart_object['kart_name']} is the ego car."
            elif chosen_type == "num_karts":
                correct_statement = f"There are {len(kart_objects)} karts in the scene."
            elif chosen_type == "track_name":
                correct_statement = f"The track is {track_name}."
            elif chosen_type in ["front_of_ego", "behind_ego", "right_of_ego", "left_of_ego"]:
                valid_karts = []
                for kart in other_karts:
                    kart_x, kart_y = kart['center']
                    ego_x, ego_y = center_kart_object['center']
                    if chosen_type == "front_of_ego" and kart_y < ego_y:
                        valid_karts.append(kart)
                    elif chosen_type == "behind_ego" and kart_y > ego_y:
                        valid_karts.append(kart)
                    elif chosen_type == "right_of_ego" and kart_x > ego_x:
                        valid_karts.append(kart)
                    elif chosen_type == "left_of_ego" and kart_x < ego_x:
                        valid_karts.append(kart)

                if valid_karts:
                    target_kart = random.choice(valid_karts)
                    relative_pos_map = {
                        "front_of_ego": "in front of",
                        "behind_ego": "behind",
                        "right_of_ego": "right of",
                        "left_of_ego": "left of"
                    }
                    correct_statement = f"{target_kart['kart_name']} is {relative_pos_map[chosen_type]} the ego car."

    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
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
    #def generate_train_caption_data(data_folder: str):
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
                for _ in range(3):  # Generate 3 captions for each view_index
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

def generate_correct_caption(info_path: str, view_index: int) -> str:
    """
    Generates a single, correct caption by randomly selecting one of the three
    specified question types: ego car, counting, or track name.
    """
    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        return "Error: Info file not found."

    track_name = info_data.get('track', 'Unknown Track')
    kart_objects = extract_kart_objects(info_path, view_index, img_width=1280, img_height=720, min_box_size=10)

    # List of possible correct statements based on your input
    correct_statements = []

    # 1. Ego car statement
    center_kart_object = next((k for k in kart_objects if k.get('is_center_kart')), None)
    if center_kart_object:
        correct_statements.append(f"{center_kart_object['kart_name']} is the ego car.")

    # 2. Counting statement
    correct_statements.append(f"There are {len(kart_objects)} karts in the scenario.")

    # 3. Track name statement
    if track_name and track_name != 'Unknown Track':
        correct_statements.append(f"The track is {track_name}.")

    # Randomly choose one correct statement
    if correct_statements:
        return random.choice(correct_statements)
    else:
        return "No information available."

def generate_all_possible_correct_captions(info_path: str, view_index: int) -> list[str]:
    """
    Generates a list of all possible correct captions for a given view,
    including ego car, counting, track name, and relative positions.
    """
    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        return ["Error: Info file not found."]

    kart_objects = extract_kart_objects(info_path, view_index, img_width=1280, img_height=720, min_box_size=10)
    track_name = info_data.get('track', 'unknown')

    correct_statements = []

    # 1. Ego car statement
    center_kart_object = next((k for k in kart_objects if k.get('is_center_kart')), None)
    if center_kart_object and 'kart_name' in center_kart_object:
        correct_statements.append(f"{center_kart_object['kart_name']} is the ego car.")

    # 2. Counting statement
    correct_statements.append(f"There are {len(kart_objects)} karts in the scenario.")

    # 3. Track name statement
    if track_name and track_name != 'unknown' and isinstance(track_name, str):
        correct_statements.append(f"The track is {track_name}.")

    # 4. Relative position statement - THIS IS THE FIXED PART
    if center_kart_object and 'center' in center_kart_object:
        other_karts = [k for k in kart_objects if not k.get('is_center_kart')]

        # The fix: only generate relative position captions if there are other karts.
        if other_karts:
            ego_x, ego_y = center_kart_object['center']

            for other_kart in other_karts:
                if 'center' in other_kart and 'kart_name' in other_kart:
                    kart_x, kart_y = other_kart['center']
                    kart_name = other_kart['kart_name']

                    x_pos = "left of" if kart_x < ego_x else "right of"
                    y_pos = "in front of" if kart_y < ego_y else "behind"

                    # Ensure the caption is specific and names the karts
                    correct_statements.append(f"{kart_name} is {y_pos} and {x_pos} the ego car.")

    return correct_statements

def generate_qa_entry(info_path: str, view_index: int) -> dict:
    """
    Generates a QA entry (multiple-choice) for a specific view,
    using a single correct caption from the list of possible captions.
    """
    all_correct_statements = generate_all_possible_correct_captions(info_path, view_index)

    if not all_correct_statements:
        return {
            "image_file": "",
            "candidates": ["No information available."],
            "correct_index": 0
        }

    correct_statement = random.choice(all_correct_statements)
    candidates = [correct_statement]

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
    kart_objects = extract_kart_objects(info_path, view_index, img_width=1280, img_height=720, min_box_size=10)
    center_kart_object = next((k for k in kart_objects if k.get('is_center_kart')), None)
    all_kart_names = info_data.get('karts', [])
    all_tracks = ["abyss", "fortmagma", "black_forest", "cornfield_crossing", "gran_paradiso_island", "hacienda", "lighthouse", "minigolf", "olivermath", "overworld", "ravenbridge_mansion", "sandtrack", "scotland", "snowmountain", "snowtuxpeak", "tutorial", "volcano_island", "xr591", "zengarden"]

    # Generate incorrect candidates based on the same five types
    # 1. Incorrect ego car statements
    if center_kart_object:
        incorrect_ego_karts = [k_name for k_name in all_kart_names if k_name != center_kart_object.get('kart_name')]
        if incorrect_ego_karts:
            candidates.append(f"{random.choice(incorrect_ego_karts)} is the ego car.")

    # 2. Incorrect total karts statements
    incorrect_num_karts_options = [n for n in range(max(0, len(kart_objects) - 2), len(kart_objects) + 3) if n != len(kart_objects)]
    if incorrect_num_karts_options:
        candidates.append(f"There are {random.choice(incorrect_num_karts_options)} karts in the scenario.")

    # 3. Incorrect track name statements
    incorrect_tracks_options = [t for t in all_tracks if t != track_name]
    if incorrect_tracks_options:
        candidates.append(f"The track is {random.choice(incorrect_tracks_options)}.")

    # 4. Incorrect relative position statements
    if center_kart_object and 'center' in center_kart_object:
        other_karts = [k for k in kart_objects if not k.get('is_center_kart')]
        if other_karts:
            target_kart = random.choice(other_karts)
            if 'kart_name' in target_kart and 'center' in target_kart:
                kart_x, kart_y = target_kart['center']
                ego_x, ego_y = center_kart_object['center']

                # Invert positions for an incorrect statement
                incorrect_x_pos = "right of" if kart_x < ego_x else "left of"
                incorrect_y_pos = "behind" if kart_y < ego_y else "in front of"

                candidates.append(f"{target_kart['kart_name']} is {incorrect_y_pos} and {incorrect_x_pos} the ego car.")

    # Pad with generic incorrect options if needed
    while len(candidates) < 5:
        generic_incorrect_options = ["The sun is setting.", "The karts are on a street.", "This is a night scene."]
        candidates.append(random.choice(generic_incorrect_options))
        candidates = list(set(candidates))

    if len(candidates) > 5:
        incorrect_candidates = [c for c in candidates if c != correct_statement]
        candidates = random.sample(incorrect_candidates, 4) + [correct_statement]

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

def generate_all_data(data_folder: str = "data/train"):
    """
    Reads a list of _info.json files from a specified folder, and generates
    corresponding _captions.json files and a single all_mc_qas.json file.

    Args:
        data_folder (str): The path to the folder containing the _info.json files.
    """
    folder_path = Path(data_folder)
    if not folder_path.exists():
        print(f"Error: The specified folder '{data_folder}' does not exist.")
        return

    info_files = list(folder_path.glob("*_info.json"))

    if not info_files:
        print(f"No _info.json files found in '{data_folder}'.")
        return

    all_mc_qas_data = []

    for info_file in info_files:
        print(f"Processing {info_file}...")

        # --- Generate _captions.json files (Training Data) ---
        captions_data = []
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            if "detections" in info_data:
                for view_index in range(len(info_data["detections"])):
                    # Get a list of all possible correct captions
                    all_correct_captions = generate_all_possible_correct_captions(str(info_file), view_index)

                    # Generate three distinct captions for training data
                    captions_to_add = random.sample(all_correct_captions, k=min(3, len(all_correct_captions)))

                    # If there are less than 3 unique captions, repeat to fill the list
                    while len(captions_to_add) < 3:
                        captions_to_add.append(random.choice(all_correct_captions))

                    for caption in captions_to_add:
                        image_file = f"{folder_path.name}/{info_file.stem.replace('_info', '')}_{view_index:02d}_im.jpg"
                        captions_data.append({
                            "image_file": image_file,
                            "caption": caption
                        })

            captions_file_path = info_file.parent / f"{info_file.stem.replace('_info', '')}_captions.json"
            with open(captions_file_path, 'w') as f:
                json.dump(captions_data, f, indent=2)
            print(f"Generated {len(captions_data)} captions for {captions_file_path.name}")

        except Exception as e:
            print(f"Error processing {info_file} for captions: {e}")

        # --- Generate data for the single all_mc_qas.json file (Validation Data) ---
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            if "detections" in info_data:
                for view_index in range(len(info_data["detections"])):
                    qa_entry = generate_qa_entry(str(info_file), view_index)
                    all_mc_qas_data.append(qa_entry)

        except Exception as e:
            print(f"Error processing {info_file} for MC QAs: {e}")

    # Save the single all_mc_qas.json file
    all_mc_qas_file_path = folder_path / "all_mc_qas.json"
    with open(all_mc_qas_file_path, 'w') as f:
        json.dump(all_mc_qas_data, f, indent=2)
    print(f"\nGenerated a single validation file with {len(all_mc_qas_data)} entries: {all_mc_qas_file_path.name}")


def main():
    fire.Fire({"check": check_caption,"generate_caption":generate_train_caption_data,"generate_validation":generate_all_mc_qas_file,"count_training_captions":count_training_captions,"generate_all_data":generate_all_data})


if __name__ == "__main__":
    main()
