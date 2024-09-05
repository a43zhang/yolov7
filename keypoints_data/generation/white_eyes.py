import cv2
import numpy as np
from PIL import Image

# Load the image using PIL and convert it to RGBA
image_path = './stitch.png'  # Replace with your image path
pil_image = Image.open(image_path).convert("RGBA")
pixels = pil_image.load()

# Convert the PIL image to an OpenCV-compatible format (BGR for display)
cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)

# Function to handle mouse clicks
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: {x}, {y}")
        
        # Define the limits (20 pixels around the selected point)
        min_x, max_x = max(0, x - 20), min(pil_image.width, x + 20)
        min_y, max_y = max(0, y - 20), min(pil_image.height, y + 20)
        
        # Iterate over the region and check for transparent pixels
        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                r, g, b, a = pixels[i, j]  # Get the RGBA values of the pixel
                if a == 0:  # If the pixel is fully transparent
                    pixels[i, j] = (255, 255, 255, 255)  # Set to white (R, G, B, A)
        
        # Convert the modified PIL image back to OpenCV format (BGRA)
        modified_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
        cv2.imshow('Image', modified_image)  # Display updated image

# Display the image in OpenCV window
cv2.imshow('Image', cv_image)

# Set the mouse callback function
cv2.setMouseCallback('Image', on_click)

# Keep the window open until the user presses a key
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the modified image if needed
pil_image.save('output_image.png')
