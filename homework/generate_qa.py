import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 600, img_height: int = 400, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Info file not found at {info_path}")
        return []

    kart_names = info_data.get('karts', [])

    if view_index >= len(info_data["detections"]):
        print(f"Error: View index {view_index} is out of range for detections in {info_path}")
        return []

    frame_detections = info_data['detections'][view_index]

    # Find the kart closest to the center of the image
    center_x_image = img_width / 2
    center_y_image = img_height / 2

    center_kart_instance_id = -1
    min_distance = float('inf')

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        if class_id == 1:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Calculate Euclidean distance from image center
            distance = np.sqrt((center_x - center_x_image)**2 + (center_y - center_y_image)**2)

            if distance < min_distance:
                min_distance = distance
                center_kart_instance_id = track_id

    kart_objects = []
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection

        if class_id == 1:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if 0 <= center_x <= img_width and 0 <= center_y <= img_height:
                kart_object = {
                    'instance_id': track_id,
                    'kart_name': kart_names[track_id] if track_id < len(kart_names) else 'Unknown',
                    'center': (center_x, center_y),
                    'is_center_kart': track_id == center_kart_instance_id,
                    'bbox': [x1, y1, x2, y2]
                }
                kart_objects.append(kart_object)

    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path, 'r') as f:
        data = json.load(f)

    # Accessing elements
    #track = data['track']
    return data.get('track', 'Unknown Track')
    #raise NotImplementedError("Not implemented")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    #raise NotImplementedError("Not implemented")
    qa_pairs = []
    kart_objects = extract_kart_objects(info_path, view_index)

    try:
        with open(info_path, 'r') as f:
            info_data = json.load(f)
    except FileNotFoundError:
        return qa_pairs

    all_kart_names = info_data.get('karts', [])
    track_name = info_data.get('track', 'Unknown Track')

    center_kart_object = next((k for k in kart_objects if k['is_center_kart']), None)

    info_path_obj = Path(info_path)
    base_name = info_path_obj.stem.replace("_info", "")
    image_file_path = f"{info_path_obj.parent.name}/{base_name}_{view_index:02d}_im.jpg"

    # if not center_kart_object:
    #     center_kart_name = "an unknown kart"
    #     # Since we can't determine the center kart, we'll generate general questions
    #     # and not the relative position questions.
    #     qa_pairs.append({
    #         'question': 'What kart is the center kart?',
    #         'answer': 'The center kart is not visible in this image.',
    #         'image_file': image_file_path
    #     })
    #     qa_pairs.append({
    #         'question': 'How many karts are there in the scenario?',
    #         'answer': str(len(all_kart_names)),
    #         'image_file': image_file_path
    #     })
    #     qa_pairs.append({
    #         'question': 'What track is this?',
    #         'answer': track_name,
    #         'image_file': image_file_path
    #     })
    #     return qa_pairs
    if center_kart_object:
        center_kart_name = center_kart_object['kart_name']
        center_kart_center_x, center_kart_center_y = center_kart_object['center']

        qa_pairs.append({
            'question': 'What kart is the ego car?',
            'answer': center_kart_name,
            'image_file': image_file_path
        })

        qa_pairs.append({
            'question': 'How many karts are there in the scenario?',
            'answer': str(len(all_kart_names)),
            'image_file': image_file_path
        })

        qa_pairs.append({
            'question': 'What track is this?',
            'answer': track_name,
            'image_file': image_file_path
        })

        karts_left_count = 0
        karts_right_count = 0
        karts_front_count = 0
        karts_behind_count = 0

        for kart in kart_objects:
            if kart['is_center_kart']:
                continue

            kart_name = kart['kart_name']
            kart_center_x, kart_center_y = kart['center']

            if kart_center_x < center_kart_center_x:
                left_right = 'left'
                karts_left_count += 1
            else:
                left_right = 'right'
                karts_right_count += 1

            qa_pairs.append({
                'question': f'Is {kart_name} to the left or right of the center kart?',
                'answer': left_right,
                'image_file': image_file_path
            })

            if kart_center_y < center_kart_center_y:
                front_behind = 'front'
                karts_front_count += 1
            else:
                front_behind = 'behind'
                karts_behind_count += 1

            qa_pairs.append({
                'question': f'Is {kart_name} in front of or behind the center kart?',
                'answer': front_behind,
                'image_file': image_file_path
            })

            qa_pairs.append({
                'question': f'Where is {kart_name} relative to the center kart?',
                'answer': f'{left_right} and {front_behind}',
                'image_file': image_file_path
            })

        qa_pairs.append({
            'question': 'How many karts are to the left of the center kart?',
            'answer': str(karts_left_count),
            'image_file': image_file_path
        })

        qa_pairs.append({
            'question': 'How many karts are to the right of the center kart?',
            'answer': str(karts_right_count),
            'image_file': image_file_path
        })

        qa_pairs.append({
            'question': 'How many karts are in front of the center kart?',
            'answer': str(karts_front_count),
            'image_file': image_file_path
        })

        qa_pairs.append({
            'question': 'How many karts are behind the center kart?',
            'answer': str(karts_behind_count),
            'image_file': image_file_path
        })

        return qa_pairs




def check_qa_pairs(info_path: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_path)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    print(" image_file ", image_file)

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_path)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_path, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


    """
    Usage Example: Visualize QA pairs for a specific file and view:
       python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0
    
    You probably need to add additional commands to Fire below.
    """
def generate_dataset_file(info_path: str):
    """
    Generate question-answer pairs and write them to a JSON file.

    Args:
        info_file: Path to the info.json file.
        view_index: Index of the view to analyze.
        output_file: Path to the output JSON file.
    """
    # qa_pairs = generate_qa_pairs(info_path, view_index)
    #
    # with open(output_file, 'w') as f:
    #     json.dump(qa_pairs, f, indent=2)
    #
    # print(f"Successfully generated {len(qa_pairs)} QA pairs and saved to {output_file}")


    """
    Generate question-answer pairs for all info files in a given folder.

    Args:
        data_folder: Path to the folder containing info.json files.
    """
    folder_path = Path(info_path)
    info_files = list(folder_path.glob("*_info.json"))

    if not info_files:
        print(f"No info.json files found in '{info_path}'.")
        return

    for info_file in info_files:
        print(f"Processing {info_file}...")
        try:
            with open(info_file, 'r') as f:
                info_data = json.load(f)

            output_file_name = f"{info_file.stem.replace('_info', '')}_qa_pairs.json"
            output_path = folder_path /"test"/ output_file_name

            # The 'detections' array is a list of detections for each view.
            # We need to loop through each view to generate QA pairs.
            all_qa_pairs = []
            for view_index in range(len(info_data["detections"])):
                qa_pairs_for_view = generate_qa_pairs(str(info_file), view_index)
                all_qa_pairs.extend(qa_pairs_for_view)

            with open(output_path, 'w') as f:
                json.dump(all_qa_pairs, f, indent=2)

            print(f"Successfully generated {len(all_qa_pairs)} QA pairs and saved to {output_path}")

        except Exception as e:
            print(f"Error processing {info_file}: {e}")

"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

Usage Example: Generate a JSON dataset file for a folder:
   python generate_qa.py generate_dataset_file --data_folder ../data/train
"""


def main():
    fire.Fire({"check": check_qa_pairs,"generate": generate_qa_pairs, "generate_dataset":generate_dataset_file})


if __name__ == "__main__":
    main()
