import argparse
import cv2
import numpy as np
import random
from PIL import Image
import os
import sys

from turtle_keypoints_data.generation.dataset_generation import click_keypoints

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.general import segments2boxes

IMAGE_SIZE = 256

def random_paste(background_image, turtle_image, min_scale=0.25, max_scale=0.65):
    """
    Randomly scales and pastes the turtle image onto the background image
    """

    w, h = turtle_image.size
    # first, we will randomly downscale the turtle image
    new_w = int(random.uniform(min_scale, max_scale) * w)
    new_h = int(random.uniform(min_scale, max_scale) * h)
    rotation_angle = 0#random.uniform(0,360)
    turtle_image = turtle_image.rotate(rotation_angle, expand=True)
    rotate_resize = turtle_image.size
    resized_turtle_image = turtle_image.resize((new_w, new_h))

    # second, will randomly choose the locations where to paste the new image
    kpt = click_keypoints(background_image)
    start_w, start_h = kpt[0]
    # third, will create the blank canvas of the same size as the original image
    canvas_image = Image.new('RGBA', (w, h))

    # and paste the resized turtle onto it, preserving the mask
    canvas_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)

    # Turtle image is of mode RGBA, while background image is of mode RGB;
    # `.paste` requires both of them to be of the same type.
    background_image = background_image.copy().convert('RGBA')
    # finally, will paste the resized turtle onto the background image
    background_image.paste(resized_turtle_image, (start_w, start_h), resized_turtle_image)
    
    return background_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background_image', type=str)
    parser.add_argument('--target_image', type=str, default='turtle.png', help='target image')
    opt = parser.parse_args()

    turtle_image = Image.open(f'./{opt.target_image}')
    bg_image = Image.open(f'./{opt.background_image}')

    # to create the training set, we will resize the turtle image to 256x256
    turtle_image_256x256 = turtle_image.resize((256, 256))
    # bg_image = bg_image.crop((0,200,600,1000))
    bg_image = bg_image.resize((256, 256))
    # paste the turtle onto background image
    aug_image = random_paste(bg_image.copy(), turtle_image_256x256.copy())

    aug_image.convert('RGB').save(f"output.jpg")