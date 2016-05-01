import cv2
from motion_tracker.utils.image_cropping import Coords, crop_and_resize

def show_img(window_name, img, msec_to_show_for=1500):
    '''Display an image on the screen for fixed length of milliseconds'''

    cv2.imshow(window_name, img)
    cv2.waitKey(msec_to_show_for)
    cv2.destroyWindow(window_name)


def add_box(img, box_coords):
    '''Add a visual blue box corresponding to box_coords'''

    img_copy = img.copy() # Drawing is inplace. Draw on copy to protect orginal
    cv2.rectangle(img_copy,
                  pt1=(box_coords.x0, box_coords.y0),
                  pt2=(box_coords.x1, box_coords.y1),
                  color=(255, 100, 0),
                  thickness=2)
    return img_copy


def show_stages_of_random_crop(img, box_coords, output_width=256, output_height=256):
    '''Display original image, the first training image, and the random crop image'''

    # Show raw image with bounding box
    img = add_box(img, box_coords)
    show_img("Original", img)

    # First training image (just resized from raw image)
    img_coords = Coords(0, 0, img.shape[1], img.shape[0])
    start_img, start_box = crop_and_resize(img, img_coords, box_coords,
                                           output_width, output_height,
                                           random_crop=False)
    start_img = add_box(start_img, start_box)
    show_img("First Training Image", start_img)

    # Second training image
    final_img, final_box_coords = crop_and_resize(img, img_coords, box_coords,
                                                  output_width, output_height)
    add_box(final_img, box_coords)
    show_img("Random Crop", final_img)

def show_single_stage(img, box_coords, title="Happy Dance Image"):
    """Show an image with its surrounding bounding box

    Args:
    ----
        img: np.ndarray
        box_coords: np.ndarray
    """

    bb_coords = Coords(box_coords[0], box_coords[1],
                       box_coords[2], box_coords[3])
    img_w_bb = add_box(img, bb_coords)
    show_img(title, img_w_bb)
