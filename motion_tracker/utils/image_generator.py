import pandas as pd
import numpy as np
import cv2
from image_cropping import Coords, crop_and_resize
from crop_vis import show_single_stage

def master_generator(crops_per_image = 10, batch_size=50,
                     output_width = 256, output_height = 256):
    """Generator yielding dictionary of image and bounding boxes.

    Args:
    ----
        crops_per_image: int
            Number of random crops of the current frame to take.
            Corresponds to both k3 and k4 in the paper, with a value of 10.
        batch_size: int
            Number of items in list to be returned.
        output_width: int
            Width in pixels of output images for start and ending image
        output_height: int
            Height in pixels of output images for start and ending image

    """

    generators_to_draw_from = [imagenet_generator(crops_per_image)] #,
                               # alov_gen = alov_generator(crops_per_image)]
    output_fields = ['start_img', 'start_box', 'end_img', 'end_box']

    while True:
        my_output = {field: np.zeros(shape = [0, output_width, output_height])
                        for field in output_fields}

        # Check count of images already in batch is less than batch_size.
        while my_output['start_img'].shape[0] < batch_size:
            for img_source in generators_to_draw_from:
                subsource_output = next(img_source)
            for field in output_fields:
                # Add outputs from subsource
                my_output[field] = np.concatenate(my_output[field],
                                                  subsource_output[field])
                # Limit output arrays from being larger than batch_size
                my_output[field] = my_output[field][:batch_size]
        yield my_output

def imagenet_generator(crops_per_image, batch_size=50,
                       output_width = 256, output_height = 256):
    """Generator yielding dictionary of image and bounding boxes.

    Args:
    ----
        crops_per_image: int
            Number of random crops of the current frame to take.
            Corresponds to both k3 and k4 in the paper, with a value of 10.
        batch_size: int
            Number of items in list to be returned.
        output_width: int
            Width in pixels of output images for start and ending image
        output_height: int
            Height in pixels of output images for start and ending image

    """

    raw_image_dir = 'data/imagenet/images/'
    img_metadata = pd.read_csv('work/imagenet/parsed_bb2.csv')

    output_fields = ['start_img', 'start_box', 'end_img', 'end_box']

    while True:
        my_output = {field: np.zeros(shape = [0, output_width, output_height])}
        while my_output[start_imgs].shape[0] < batch_size:
            img_row = img_metadata.sample(1)
            raw_img = cv2.imread(raw_image_dir + img_row.filename.values[0])
            img_coords = Coords(0, 0, img_row.width, img_row.height)
            box_coords = Coords(img_row.x0, img_row.y0,
                                img_row.x1, img_row.y1)

            start_img, start_box = crop_and_resize(raw_img, img_coords, box_coords,
                                                   output_width, output_height,
                                                   random_crop=False)
        for image_crop in range(crops_per_image):
            end_img, end_box = crop_and_resize(raw_img, img_coords, box_coords,
                                       output_width, output_height,
                                       random_crop=True)

        my_output['start_img'] = np.concatenate(my_output['start_img'], start_img)
        my_output['start_box'] = np.concatenate(my_output['start_box'], start_img)
        my_output['end_img'] = np.concatenate(my_output['end_img'], end_img)
        my_output['end_box'] = np.concatenate(my_output['end_box'], end_img)

        # Limit output arrays from being larger than batch_size
        for field in output_fields:
            my_output[field] = my_output[field][:batch_size]
        yield my_output

# --- DID NOT GET BELOW THIS LINE
def alov_generator(num_video_crops, output_width = 256, output_height = 256):
    """Generator yielding alov pair frames and bounding boxes.

    The training procedure used takes the two succesive frames,
    labeled previous and current. The previous frame is yielded
    with it's original bounding box, whereas ten random croppings
    of the the current are taken and yielded.

    Args:
    ----
        num_video_crops: int
            Number of random croppings to take of the current frame.

    Outputs:
    -------
        imgs_lst: list of images
        boxes_lst: list of bounding box coords
    """

    raw_image_dir = 'work/alov/images/'
    img_metadata = pd.read_csv('work/alov/parsed_bb.csv')

if __name__ == '__main__':
    mas_gen = master_generator()
    imgs, boxes = next(mas_gen)
    for img, box in zip(imgs[0], boxes[0]):
        show_single_stage(img, box)
