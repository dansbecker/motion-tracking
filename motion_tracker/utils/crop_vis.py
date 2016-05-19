import cv2
from motion_tracker.utils.image_cropping import Coords, crop_and_resize

def show_stages_of_random_crop(img, box_coords, output_width=256, output_height=256):
    """Display original image, the first training image, and the random crop image

    Args:
    ----
        img: np.ndarray
        box_coords: Coords object
        output_width (optional): int
        output_height (optional): int
    """

    # Show raw image with bounding box
    show_img(img, box_coords.as_array())

    # First training image (just resized from raw image)
    img_coords = Coords(0, 0, img.shape[1], img.shape[0])
    start_img, start_box_coords = crop_and_resize(img, img_coords, box_coords,
                                           output_width, output_height,
                                           random_crop=False)
    show_img(start_img, start_box_coords.as_array())

    # Second training image
    final_img, final_box_coords = crop_and_resize(img, img_coords, box_coords,
                                                  output_width, output_height)
    show_img(final_img, final_box_coords.as_array())


def show_img(img, boxes=None, window_name="Happy Dance Image", msec_to_show_for=1500):
    """Show an image, potentially with surrounding bounding boxes

    Args:
    ----
        img: np.ndarray
        boxes (optional): dct of bounding boxes where the keys hold the name (actual
            or predicted) and the values the coordinates of the boxes
        window_name (optional): str
        msec_to_show_for (optioanl): int
    """

    img_copy = img.copy() # Any drawing is inplace. Draw on copy to protect original.
    if boxes:
        color_dct = {'actual': (125, 255, 0), 'predicted': (0, 25, 255)}
        for box_type, box_coords  in boxes.items():
            cv2.rectangle(img_copy,
                          pt1=(box_coords[0], box_coords[1]),
                          pt2=(box_coords[2], box_coords[3]),
                          color=color_dct[box_type],
                          thickness=2)
    cv2.imshow(window_name, img_copy)
    cv2.waitKey(msec_to_show_for)
    cv2.destroyWindow(window_name)
