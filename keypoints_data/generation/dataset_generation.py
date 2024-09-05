import argparse
import cv2
import numpy as np
import random
from PIL import Image
from pathlib import Path
import torch
from pytorch_pretrained_biggan import (
    BigGAN,
    truncated_noise_sample,
    convert_to_images,
    one_hot_from_int,
)
import torchvision.transforms as transforms
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.general import segments2boxes

IMAGE_SIZE = 256

KEYPOINTS = [
    (235, 119), #eye
    (120, 80), #center vertex
    (114, 253), #bottom right
    (3, 132), #bottom left
    (45, 14), #top left
    (187, 1), #top right
]

def BigGan_generate(batch_size = 10, desired_num_im = 20):
    """
    Generate background images with generative adversarial network "BigGAN" pre-trained on ImageNet
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = BigGAN.from_pretrained('biggan-deep-256').to(device).eval()
    truncation = 0.4
    background_images = []
    num_im = 0

    while num_im < desired_num_im:
        # Imagenet 1000 categories
        class_vector = torch.from_numpy(one_hot_from_int(np.random.randint(0, 1000, size=batch_size), batch_size=batch_size)).to(device)
        noise_vector = torch.from_numpy(truncated_noise_sample(truncation=truncation, batch_size=batch_size)).to(device)
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation).cpu()
            candidate_images = convert_to_images(output)
        background_images.extend(candidate_images)
        num_im += len(candidate_images)
        print(num_im)

    del model
    torch.cuda.empty_cache()
    return background_images


def paste_randomly(background_image, target_image, scale_range=[0.3, 0.7], keypoints=None):
    """
    Randomly rotates, scales, and translates the target image before pasting onto the background image
    """
    
    # Rotate
    rotation_angle = random.uniform(0,360)
    target_image = target_image.rotate(rotation_angle, expand=True)
    rotate_resize = target_image.size

    # Downscale
    w, h = target_image.size
    scale_factor = np.random.uniform(scale_range, 2)
    scaled_w, scaled_h = int(scale_factor[0] * w), int(scale_factor[1] * h)
    resized_target_image = target_image.resize((scaled_w, scaled_h))

    # Randomize translation
    translated_w, translated_h = random.randint(0, w - scaled_w), random.randint(0, h - scaled_h)

    # Paste the resized target onto blank canvas
    canvas_image = Image.new('RGBA', (w, h))
    canvas_image.paste(resized_target_image, (translated_w, translated_h), resized_target_image)

    # Both images need to be RGBA for pasting
    background_image = background_image.convert('RGBA')
    background_image.paste(resized_target_image, (translated_w, translated_h), resized_target_image)

    # Do the same to keypoints
    if keypoints is not None:
        # Rotate
        angle_rad = np.deg2rad(rotation_angle)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])
        center = np.array([w / 2, h / 2])
        keypoints = np.dot(keypoints - center, rotation_matrix) + np.array(rotate_resize)/2
        # Scale
        keypoints[:, 0] *= scaled_w/rotate_resize[0]
        keypoints[:, 1] *= scaled_h/rotate_resize[1]
        # Translate
        keypoints += np.array([translated_w, translated_h])

    return background_image, canvas_image, keypoints

def click_keypoints(target_image_256x256):
    keypoints = []
    # Mouse callback function
    def select_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            keypoints.append([x, y])
            print(f"Point selected: ({x}, {y})")

    # Convert PIL image to a format that OpenCV can display
    cv_image = np.array(target_image_256x256)  # Convert to numpy array
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR format for OpenCV
    cv2.imshow('Select Points', cv_image)
    cv2.setMouseCallback('Select Points', select_point)

    print("Click on the image to select points. Press 'q' to quit.")
    while True:
        cv2.imshow('Select Points', cv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Close the OpenCV window
    cv2.destroyAllWindows()
    print("Selected points:", keypoints)
    return keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_im', type=int, default=20, help='number of images to generate')
    parser.add_argument('--image_name', type=str, default='target.png', help='target image')
    parser.add_argument('--select_points', action='store_true', help='select new keypoints')
    opt = parser.parse_args()

    # 4-channel RGB+Alpha image of size 2394x1800
    target_image = Image.open(f'./{opt.image_name}')

    # to create the training set, we will resize the target image to 256x256
    target_image_256x256 = target_image.resize((256, 256))

    if opt.select_points:
        keypoints = click_keypoints(target_image_256x256)
    else:
        keypoints = KEYPOINTS

    base_dir = Path(__file__).resolve().parent.parent
    imgs_dir = base_dir / "images"
    labels_dir = base_dir / "labels"

    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (imgs_dir / "train").mkdir(parents=True, exist_ok=True)
    (imgs_dir / "val").mkdir(parents=True, exist_ok=True)
    (imgs_dir / "test").mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    print("Creating background images")
    background_images = BigGan_generate(desired_num_im=opt.num_im)

    train_val_test_split = [7, 2, 1]
    total_images = len(background_images)
    train_threshold = int(train_val_test_split[0] / sum(train_val_test_split) * total_images)
    val_threshold = int((train_val_test_split[0] + train_val_test_split[1]) / sum(train_val_test_split) * total_images)

    print("Saving data")

    train_dir = "train"
    val_dir = "val"
    test_dir = "test"

    tensor_transform = transforms.ToTensor()
    paths = {train_dir: [], val_dir: [], test_dir: []}
    for idx, background_image in enumerate(background_images):
        # paste the target onto background image
        aug_image, aug_mask, aug_keypoints = paste_randomly(background_image.copy(), target_image_256x256.copy(), keypoints=np.array(keypoints, dtype=np.float64))

        # Normalize and put mask into pixel coordinates
        mask_points = torch.stack(torch.where(tensor_transform(aug_mask)[-1] > 0), 1) / IMAGE_SIZE
        mask_points = mask_points[:, [1, 0]]
        # get bbox in xywh from points
        box = segments2boxes([mask_points])[0]

        if aug_keypoints is not None:
            aug_keypoints = torch.tensor(aug_keypoints)
            aug_keypoints = aug_keypoints / IMAGE_SIZE
        else:
            raise Exception("No keypoints?")

        if idx < train_threshold:
            split = train_dir
        elif idx < val_threshold:
            split = val_dir
        else:
            split = test_dir
        
        if split != test_dir:
            with open(labels_dir / split / f"{idx}.txt", 'w') as f:
                line = f"{0} " + " ".join(map(str, box.tolist())) + " " + " ".join(map(str, aug_keypoints.flatten().tolist()))
                f.write(line + "\n")
        aug_image.convert('RGB').save(imgs_dir / split / f"{idx}.jpg")
        paths[split].append("./" + str(Path("images") / split / f"{idx}.jpg"))

    for txt_file in [train_dir, val_dir, test_dir]:
        with open(base_dir / f"{txt_file}.txt", 'w') as f:
            f.writelines(line + "\n" for line in paths[txt_file])