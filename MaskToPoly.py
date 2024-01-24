import os
import cv2
import numpy as np

# Read the labelmap file and skip the header line
labelmap_path = './labelmap.txt'
with open(labelmap_path, 'r') as labelmap_file:
    lines = labelmap_file.readlines()[1:]  # Skip the first line

# Process the rest of the lines
labelmap_data = []
for line in lines:
    parts = line.strip().split(':')
    label = parts[0]
    rgb_values = list(map(int, parts[1].split(',')))
    labelmap_data.append((tuple(reversed(rgb_values)), label))

# Sort the list so that "background" is at the bottom
labelmap_data.sort(key=lambda x: x[1] == 'background')

# Create a mapping of labels to numbers
label_to_number = {label: str(i) for i, (rgb_values, label) in enumerate(labelmap_data)}

# Modified color mapping with BGR values
color_mapping = {rgb_values: label_to_number[label] for rgb_values, label in labelmap_data}

# Print the color_mapping
print("Color Mapping:")
for rgb_values, label in labelmap_data:
    print(f"{rgb_values}: {label_to_number[label]}")

input_dir = './maskval'
output_dir = './val'

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)

    # Load the color image
    color_image = cv2.imread(image_path)

    H, W, _ = color_image.shape

    # Create a bitmask based on color mapping
    bitmask = np.zeros((H, W), dtype=np.uint8)
    for color, index in color_mapping.items():
        lower_bound = np.array(color, dtype=np.uint8)
        upper_bound = np.array(color, dtype=np.uint8)
        matching_pixels = cv2.inRange(color_image, lower_bound, upper_bound)
        bitmask[matching_pixels > 0] = index

    # Find contours for all classes combined
    combined_mask = (bitmask < (len(labelmap_data))-1).astype(np.uint8)  # Exclude class 6
    contours_combined, hierarchy_combined = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons for all classes combined
    polygons_combined = []
    for cnt in contours_combined:
        if cv2.contourArea(cnt) > 200:
            polygon_combined = []
            for point in cnt:
                x, y = point[0]
                # Normalize the coordinates and append to the polygon
                polygon_combined.append((x / W, y / H))
            polygons_combined.append(polygon_combined)

    # Find the class of each polygon based on its position
    class_mapping = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}
    for class_index in range(len(set(color_mapping.values())) - 1):  # Exclude class 6
        class_mask = (bitmask == class_index).astype(np.uint8)
        contours, hierarchy = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append((x / W, y / H))

                # Check if the polygon matches any class and update the mapping
                if not class_mapping[class_index]:
                    class_mapping[class_index] = polygon

    # Save the polygons to a text file for all classes combined
    output_file_path_combined_polygons = os.path.join(output_dir, '{}.txt'.format(j[:-4]))
    with open(output_file_path_combined_polygons, 'w') as f:
        for polygon_combined in polygons_combined:
            for p_, p in enumerate(polygon_combined):
                if p_ == 0:
                    # Find the class of the polygon based on its position
                    for class_index, class_polygon in class_mapping.items():
                        if class_polygon and p in class_polygon:
                            f.write('{} {} '.format(class_index, ' '.join(map(str, p))))
                            break
                else:
                    f.write('{} '.format(' '.join(map(str, p))))
                if p_ == len(polygon_combined) - 1:
                    f.write('\n')
