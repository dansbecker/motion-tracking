import cv2
from code.data_setup.image_cropping import Coords, update_box_after_resize, crop_and_resize

def show_img(window_name, img, msec_to_show_for=2000):
    cv2.imshow(window_name, img)
    cv2.waitKey(msec_to_show_for)
    cv2.destroyWindow(window_name)

def add_box(img, box_coords):
    cv2.rectangle(img,
                  pt1=(box_coords.x0, box_coords.y0),
                  pt2=(box_coords.x1, box_coords.y1),
                  color=blue,
                  thickness=2)
    return img

def show_stages_of_random_crop(img, box_coords, output_width=256, output_height=256):
    blue = (255, 100, 0)
    original_img = img.copy()
    img_coords = Coords(0, 0, img.shape[1], img.shape[0])

    # Show raw image with bounding box
    img = add_box(img, box_coords)
    show_img("Original", img)

    # First training image (just resized from raw image)
    first_training_img = cv2.resize(img, (output_width, output_height))
    first_training_box = update_box_after_resize(box_coords, img_coords,
                                               output_width, output_height)
    first_training_img = add_box(first_training_img, first_training_box)
    show_img("First Training Image", first_training_img)

    # Second training image
    final_img, final_box_coords = crop_and_resize(img, img_coords, box_coords,
                                                  output_width, output_height)
    add_box(final_img, box_coords)
    show_img("Random Crop", final_img)
