"""A script for building image generators for net modeling."""

import pandas as pd
import numpy as np
import cv2
from image_cropping import Coords, crop_and_resize
from crop_vis import show_single_stage 

def master_generator(num_video_crops=10, num_image_crops=10, batch_size=50): 
    """Generator yielding a batch of images. 

    Args: 
    ----
        num_video_crops: int
            Number of random crops of the current frame to take. 
            Denoted as k3 in the paper, with a value of 10. 
        num_image_crops: int 
            Number of random crops of the randomly sampled image 
            to take. Dentoed as k4 in the paper, with a value of 10. 
        batch_size: int
            Number of items in list to be returned. 
    """
    # To switch witch training images we draw from first in 
    # a batch. 
    imagenet = True
    imagenet_gen = imagenet_generator(num_image_crops)

    while True: 
        
        out_imgs = []
        out_boxes = []
        video_pairs = 0
        images = 0
        num_obs = 0
        # Might need to add in some check or something in the 
        # case that the num_video_crops and num_image_crops can't
        # logically (and in an alternating way) add up to the 
        # batch size. This while loop figures out how many pairs 
        # of video frames to grab and how many images to grab. 
        while num_obs < batch_size: 
            if imagenet: 
                images += 1
                num_obs += num_video_crops
            else: 
                video_pairs += 1
                num_obs += num_image_crops
            imagenet = not imagenet

        print(images, video_pairs)
        
        # Now grab the actual images and video based off the number
        # of pairs and sets of random croppings we need. 
        for idx in range(images): 
            imgs, boxes = next(imagenet_gen)
            out_imgs.append(imgs)
            out_boxes.append(boxes)

        '''
        for idx in range(video_pairs): 
            imgs, boxes = next(alov_gen)
            out_imgs.append(imgs)
            out_boxes.append(boxes)
        '''

        yield out_imgs, out_boxes
            
def imagenet_generator(num_image_crops):
    """Generator yielding images and bounding boxes. 

    Args: 
    ----
        num_image_crops: int
            Number of image crops to take of each randomly drawn 
            image. 

    Outputs:
    -------
        imgs_lst: list of images
        boxes_lst: list of bounding box coords
    """
    
    raw_image_dir = 'data/imagenet/images/'
    img_metadata = pd.read_csv('work/imagenet/parsed_bb2.csv')
    output_width = 256
    output_height = 256

    while True:
        imgs_lst = []
        boxes_lst = []
        img_row = img_metadata.sample(1)
        
        # Make sure to read in the image only once, and calculate 
        # the img and box coords only once. We don't need to calculate
        # those for every random cropping. 
        raw_img = cv2.imread(raw_image_dir + img_row.filename.values[0])
        img_coords = Coords(0, 0, img_row.width, img_row.height)
        box_coords = Coords(img_row.x0, img_row.y0, 
                img_row.x1, img_row.y1)
        
        # Put the original as the first thing item in the generator. 
        # We might have to separate this out somehow, since it's going
        # to a different part of the net from the 10 random croppings 
        # (I think?). 
        img, box = crop_and_resize(raw_img, 
                img_coords, box_coords, output_width,
                output_height, random_crop=False)
        imgs_lst.append(np.array(img))
        boxes_lst.append(np.array(box.as_array()))

        # Now take the right number of random croppings, and append
        # those before yielding them. 
        for image_crop in range(num_image_crops): 
            img, box = crop_and_resize(raw_img, 
                    img_coords, box_coords, output_width,
                    output_height, random_crop=True)

            imgs_lst.append(np.array(img))
            boxes_lst.append(np.array(box.as_array()))

        yield imgs_lst, boxes_lst

if __name__ == '__main__': 
    mas_gen = master_generator() 
    imgs, boxes = next(mas_gen)
    for img, box in zip(imgs[0], boxes[0]): 
        show_single_stage(img, box)
