import os
import cv2
import numpy as np
import argparse

def draw_polygon(image, points, color=(0, 255, 0), thickness=2):
    """
    Draw a polygon on the image using the given points.
    
    Args:
        image: The image on which to draw the polygon.
        points: A list of (x, y) tuples representing the polygon vertices.
        color: The color of the polygon line.
        thickness: The thickness of the polygon line.
    """
    # Convert normalized points back to image coordinates
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Draw the polygon
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

def load_labels(label_path):
    """
    Load the labels from the txt file.
    
    Args:
        label_path: Path to the label txt file.
    
    Returns:
        A list of polygons, where each polygon is a list of (x, y) tuples.
    """
    polygons = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            bbox = parts[1:5]
            coordinates = parts[5:]
            
            # Convert flat list to list of (x, y) tuples
            points = [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]
            polygons.append(points)
    
    return polygons, bbox

import cv2

def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2):
    """
    Draws a bounding box on the image.
    
    Args:
        image: The image on which to draw the bounding box (as a NumPy array).
        bbox: A tuple or list in the format (x_center, y_center, width, height) 
              with normalized coordinates (relative to image width and height).
        color: The color of the bounding box in BGR format (default is green).
        thickness: The thickness of the bounding box lines (default is 2).
        
    Returns:
        image: The image with the bounding box drawn on it.
    """
    img_height, img_width = image.shape[:2]
    
    # Convert normalized coordinates to absolute pixel values
    x_center = int(bbox[0] * img_width)
    y_center = int(bbox[1] * img_height)
    width = int(bbox[2] * img_width)
    height = int(bbox[3] * img_height)
    
    # Calculate the top-left and bottom-right corners of the bounding box
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    # Draw the rectangle on the image
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    return image


def main(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image {image_path} not found.")
        return
    
    # Load the labels
    polygons, bbox = load_labels(label_path)
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Draw each polygon on the image
    for polygon in polygons:
        # Convert normalized coordinates to pixel coordinates
        pixel_polygon = [(int(x * width), int(y * height)) for x, y in polygon]
        draw_polygon(image, pixel_polygon)
    
    draw_bbox(image, bbox)

    # Display the image
    cv2.imshow('Image with Polygons', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--label', type=str)
    opt = parser.parse_args()

    # /home/alan/ML/yolov7/coco/images/train2017/000000001374.jpg
    main(opt.image, opt.label)
