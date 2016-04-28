import cv2
import os
import numpy as np
import pandas as pd
from numpy.random import laplace

class Coords(object):
    '''Utility class to hold coords and automatically calculate height, width and center

    x0, y0, x1 and y1 are all integers
    '''

    def __init__(self, x0, y0, x1, y1):
        self.x0 = int(x0) # convert to int here. Otherwise shows up in many places
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.height = self.y1 - self.y0
        self.width = self.x1 - self.x0
        self.x_center = (self.x0 + self.x1) / 2
        self.y_center = (self.y0 + self.y1) / 2
    def __repr__(self):
        return(", ".join([str(i) for i in (self.x0, self.y0, self.x1, self.y1)]))
    def as_array(self):
        return np.array([self.x0, self.y0, self.x1, self.y1])

def get_cropping_params(bb_loc_laplace_b_param = 0.2,
                    bb_size_laplace_b_param = 0.06,
                    bbox_shrinkage_limit = 0.6,
                    bbox_expansion_limit = 1.4):
    '''Get random draws for parameters involved in random crops. Output those params

    Args:
    ----
        bb_loc_laplace_b_param: Scale parameter in the laplace distribution that
            determines where to center the hypothetical bounding box used to
            create crop. Applied for draws of both x and y shifts.
        bb_size_laplace_b_param" Scale parameter in the laplace distribution that
            determines the size of the hypothetical bounding box used to create
            crop. Applied for draws of both x and y scaling.
        bbox_shrinkage_limit: The minimum size in each dimension of hypothetical
            bounding box for cropping, as a fraction of previous bounding box
            size in that dimension
        bbox_expansion_limit: Maximum size in each dimension of hypothetical bounding
            box for cropping, as a fraction of previous bounding box size in that
            dimension.

    Output:
    ------
        x_center_shift: fraction of box width to shift center of bbox for next crop
        y_center_shift: fraction of box height to shift center of bbox for next crop
        x_size_shift: bbox width used to make crop, as frac of previous bbox width
        y_size_shift: bbox height used to make crop, as frac of previous bbox height
    '''

    x_center_shift = laplace(scale=bb_loc_laplace_b_param) #delta_x in paper
    y_center_shift = laplace(scale=bb_loc_laplace_b_param) #delta_y in paper
    x_size_shift = (1 - laplace(scale=bb_size_laplace_b_param))
    x_size_shift = np.clip(x_size_shift, bbox_shrinkage_limit, bbox_expansion_limit)
    y_size_shift = (1 - laplace(scale=bb_size_laplace_b_param))
    y_size_shift = np.clip(y_size_shift, bbox_shrinkage_limit, bbox_expansion_limit)
    return x_center_shift, y_center_shift, x_size_shift, y_size_shift

def update_box_after_crop(box_coords, crop_coords):
    '''Calculate coords of correct bounding box after random crop and resizing

    Args:
    -----
    box_coords: coordinates of correct bounding box in original/uncropped image
    crop_coords: coordinates/boundaries of newly cropped area

    Outputs:
    -------
    Coords object with coordinates of correct bounding box in cropped image
    '''

    # Translate box. Use min and max to deal with case where bounding box hits edge
    new_box_x0 = max(box_coords.x0 - crop_coords.x0, 0)
    new_box_y0 = max(box_coords.y0 - crop_coords.y0, 0)
    new_box_x1 = min(box_coords.x1 - crop_coords.x0, crop_coords.x1)
    new_box_y1 = min(box_coords.y1 - crop_coords.y0, crop_coords.y1)
    return Coords(new_box_x0, new_box_y0, new_box_x1, new_box_y1)

def update_box_after_resize(box_coords, img_coords, output_img_width=256, output_img_height=256):
    '''Calculate coords of correct bounding box to reflect resizing

    Args:
    -----
    box_coords: coordinates of correct bounding box before resizing
    crop_coords: coordinates/boundaries of image before resizing
        It doesn't matter if x0 and y0 are 0, or if these are positive denoting
        we are working with a crop. These coordinates are used only to find the
        height and width before resizing.

    Outputs:
    -------
    Coords object with coordinates of correct bounding box in resized image
    '''

    width_resize_ratio = output_img_width / (img_coords.x1 - img_coords.x0)
    height_resize_ratio = output_img_height / (img_coords.y1 - img_coords.y0)

    new_box_x0 = box_coords.x0 * width_resize_ratio
    new_box_y0 = box_coords.y0 * height_resize_ratio
    new_box_x1 = box_coords.x1 * width_resize_ratio
    new_box_y1 = box_coords.y1 * height_resize_ratio
    return Coords(new_box_x0, new_box_y0, new_box_x1, new_box_y1)

def get_random_box_to_determine_crop(box_coords):
    '''Create the hypothetical bounding box used in determining the next crop

    Args:
    -----
    box_coords: Coords object containing the location of the true bounding box
        in the original image

    Outputs:
    --------
    Coords object with the hypothetical bounding box to use for the next crop
    '''
    # though box center, height and width are calculated automatically in Coords
    # constructor, we calculate them explicitly here to find the edges
    # (i.e. to create new box Coords)
    x_shift_frac, y_shift_frac, x_size_change_frac, y_size_change_frac = get_cropping_params()
    box_for_crop_x_center = box_coords.x_center + x_shift_frac * box_coords.width
    box_for_crop_y_center = box_coords.y_center + y_shift_frac * box_coords.height
    box_for_crop_width = box_coords.width * x_size_change_frac
    box_for_crop_height = box_coords.height * y_size_change_frac

    box_for_crop_x0 = box_for_crop_x_center - 0.5 * box_for_crop_width
    box_for_crop_y0 = box_for_crop_y_center - 0.5 * box_for_crop_height
    box_for_crop_x1 = box_for_crop_x_center + 0.5 * box_for_crop_width
    box_for_crop_y1 = box_for_crop_y_center + 0.5 * box_for_crop_height
    box_for_crop = Coords(box_for_crop_x0, box_for_crop_y0, box_for_crop_x1, box_for_crop_y1)
    return box_for_crop

def get_crop_coords(box_coords, img_coords, random_crop):
    '''Calculate the boundaries for a new crop and the coords of the correct bounding box
    in that new crop.

    Args:
    -----
    box_coords: Coords object with location of bounding box in original img
    img_coords: Coords (basically the size) of original image
    random_crop: If True, create stochastic box to determine crop coords.
                 If False, create crop based on true bounding box
                 Cropped area is twice as large (and centered) on box either way

    Outputs:
    --------
    cropped_area_coords: Coords object with boundaries of the random crop
    new_box_coords: Coords object with location of correct bounding box in
        newly cropped image
    '''

    if random_crop:
        box_for_crop = get_random_box_to_determine_crop(box_coords)
    else:
        box_for_crop = box_coords

    # calc boundary coords of cropped area. Crop_img is twice as big as box_for_crop
    cropped_area_x0 = max(box_for_crop.x_center - box_for_crop.width, 0)
    cropped_area_y0 = max(box_for_crop.y_center - box_for_crop.height, 0)
    cropped_area_x1 = min(box_for_crop.x_center + box_for_crop.width, img_coords.width)
    cropped_area_y1 = min(box_for_crop.y_center + box_for_crop.height, img_coords.height)
    cropped_area_coords = Coords(cropped_area_x0, cropped_area_y0,
                                 cropped_area_x1, cropped_area_y1)

    box_after_crop = update_box_after_crop(box_coords, cropped_area_coords)
    return cropped_area_coords, box_after_crop

def read_bbox_data(raw_image_dir, parsed_bb_path='work/imagenet/parsed_bb.csv'):
    '''Return df with data about images and bounding boxes.

    Data captured corresponds to xml files obtained from ImageNet.
    '''

    # paper says they use .66, but they use search area that is twice bounding box
    # So, 0.5 may be more appropriate.
    max_box_frac_of_width =  0.66
    max_box_frac_of_height = 0.66
    images_successfully_downloaded = set(os.listdir('data/imagenet/images'))
    bbox_df = (pd.read_csv('work/imagenet/parsed_bb.csv')
                    .assign(box_height = lambda df: df.ymax - df.ymin,
                            box_width = lambda df: df.xmax - df.xmin)
                    .assign(box_frac_of_height = lambda df: df.box_height / df.height,
                            box_frac_of_width = lambda df:  df.box_width / df.width)
                    .query('filename in @images_successfully_downloaded')
                    .query('box_frac_of_height < @max_box_frac_of_height')
                    .query('box_frac_of_width < @max_box_frac_of_width'))
    return bbox_df

def crop_and_resize(img, img_coords, box_coords, output_width, output_height, random_crop=True):
    '''Return image and the bounding box after a possibly random crop and resize

    Args:
    -----
    img: ndarray of image (consistent with opencv format)
    img_coords: Coordinates of image boundaries
    box_coords: Bounding box coordinates on img
    output_width: Desired output width for resizing
    output_height: Desired output height for resizing
    random_crop: If True, create stochastic box to determine crop coords.
                 If False, create crop based on true bounding box
                 Cropped area is twice as large (and centered) on box either way
    '''

    crop_coords, box_after_crop = get_crop_coords(box_coords, img_coords, random_crop)
    cropped_img = img[crop_coords.y0:crop_coords.y1,
                      crop_coords.x0:crop_coords.x1]
    box_after_crop = update_box_after_crop(box_coords, crop_coords)
    final_img = cv2.resize(cropped_img, (output_width, output_height))
    final_box_coords = update_box_after_resize(box_after_crop, crop_coords,
                                               output_width, output_height)
    return final_img, final_box_coords

def imagenet_generator(batch_size=50):
    '''Generator yielding dictionary. Each dictionary is array of data for model

    batch_size: Number of items in list to be returned.
    '''
    raw_image_dir = 'data/imagenet/images/'
    img_metadata = read_bbox_data(raw_image_dir)
    output_width = 256
    output_height = 256

    start_img_data = []
    start_box_data = []
    next_img_data = []
    next_box_data = []

    while True:
        batch_df = img_metadata.sample(batch_size)
        for _, row in batch_df.iterrows():
            try:
                img = cv2.imread(raw_image_dir + row.filename)
                img_coords = Coords(0, 0, row.width, row.height)
                box_coords = Coords(row.xmin, row.ymin, row.xmax, row.ymax)
                start_img, start_box = crop_and_resize(img, img_coords, box_coords,
                                                       output_width, output_height,
                                                       random_crop=False)
                next_img, next_box = crop_and_resize(img, img_coords, box_coords,
                                                   output_width, output_height)
                start_img_data.append(start_img)
                start_box_data.append(start_box.as_array())
                next_img_data.append(next_img)
                next_box_data.append(next_box.as_array())
            except:
                print('Failed on ', row.filename)

        yield {'start_img': np.array(start_img_data),
               'start_box': np.array(start_box_data),
               'next_img': np.array(next_img_data),
               'next_box': np.array(next_box_data)}
