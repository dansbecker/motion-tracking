import pandas as pd
import numpy as np
import cv2
from motion_tracker.utils.image_cropping import Coords, crop_and_resize
from motion_tracker.utils.crop_vis import show_img

def get_X_y_containers(output_width=256, output_height=256):
    empty_img_accumulator = np.zeros([0, output_width, output_height, 3]).astype('uint8')
    empty_box_accumulator = np.zeros([0, 4]).astype('uint16')
    empty_X = {'start_img': empty_img_accumulator,
               'start_box': empty_box_accumulator,
               'end_img': empty_img_accumulator}
    empty_y = {'end_box': empty_box_accumulator}
    return empty_X, empty_y

def rough_normalization(img):
    return (img-100) / 60

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

    generators_to_draw_from = [imagenet_generator(batch_size=crops_per_image,
                                                  crops_per_image=crops_per_image,
                                                  output_width=output_width,
                                                  output_height=output_height),
                               alov_generator(batch_size=crops_per_image,
                                              crops_per_image=crops_per_image,
                                              output_width=output_width,
                                              output_height=output_height)]

    while True:
        X, y = get_X_y_containers(output_width, output_height)

        # Check we haven't hit batch size yet
        while X['start_img'].shape[0] < batch_size:
            for img_source in generators_to_draw_from:
                subsource_X, subsource_y = next(img_source)
                for field in X:
                    X[field] = np.concatenate([X[field], subsource_X[field]])
                    # Limit output array size if larger than batch
                    X[field] = X[field][:batch_size]
                y['end_box'] = np.concatenate([y['end_box'], subsource_y['end_box']])
                y['end_box'] = y['end_box'][:batch_size]
        # yield [X['start_img'], X['start_box'], X['end_img']], y
        y = y['end_box']
        # X['start_box'] = y
        yield [X['start_img'], X['start_box'], X['end_img']], {'x_0': y[:, 0]/output_width,
                                                               'y_0': y[:, 1]/output_height,
                                                               'x_1': y[:, 2]/output_width,
                                                               'y_1': y[:, 3]/output_height}

def set_array_dims(img, box):
    ''' Adds an additional axis of length 1 to start of imgs and boxes.

        These dimensionality changes are necessary for merging to accumulators
        that have observations in the first axis
    '''

    out_img = img[np.newaxis].astype('uint8')
    out_box = box.as_array()[np.newaxis]
    return out_img, out_box

def imagenet_generator(crops_per_image=10, batch_size=50,
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

    while True:
        X, y = get_X_y_containers(output_width, output_height)
        while X['start_img'].shape[0] < batch_size:
            try:
                img_row = img_metadata.sample(1)
                raw_img = cv2.imread(raw_image_dir + img_row.filename.values[0])
                img_coords = Coords(0, 0, img_row.width, img_row.height)
                box_coords = Coords(img_row.x0, img_row.y0,
                                    img_row.x1, img_row.y1)

                start_img, start_box = crop_and_resize(raw_img, img_coords, box_coords,
                                                       output_width, output_height,
                                                       random_crop=False)
                start_img, start_box = set_array_dims(start_img, start_box)
                for image_crop in range(crops_per_image):
                    end_img, end_box = crop_and_resize(raw_img, img_coords, box_coords,
                                                       output_width, output_height,
                                                       random_crop=True)
                    end_img, end_box = set_array_dims(end_img, end_box)
                    X['start_img'] = np.concatenate([X['start_img'], start_img])
                    X['start_box'] = np.concatenate([X['start_box'], start_box])
                    X['end_img'] = np.concatenate([X['end_img'], end_img])
                    y['end_box'] = np.concatenate([y['end_box'], end_box])

                # Limit output array sizes if larger than batch
                for field in X:
                    X[field] = X[field][:batch_size]
                y['end_box'] = y['end_box'][:batch_size]
            except:
                pass
        # y = y['end_box']
        yield X, y
        # yield [X['start_img'], X['start_box'], X['end_img']], {'x_0': y[:, 0]/output_width,
                                                               # 'y_0': y[:, 1]/output_height,
                                                            # 'x_1': y[:, 2]/output_width,
                                                               # 'y_1': y[:, 3]/output_height}
        # yield [X['start_img'], X['start_box'], X['end_img']], y


def alov_generator(crops_per_image=10, batch_size=50,
                       output_width = 256, output_height = 256):
    """Generator yielding dictionary of image and bounding boxes.

    Args:
    ----
        crops_per_image: int
            Number of random crops of the current frame to take.
            Corresponds to both k3 and k4 in the paper, with a value of 10.
        batch_size: in
            Number of items in list to be returned.
        output_width: int
            Width in pixels of output images for start and ending image
        output_height: int
            Height in pixels of output images for start and ending image

    """

    raw_image_dir = 'work/alov/images/'
    img_metadata = pd.read_csv('work/alov/parsed_bb.csv')

    while True:
        X, y = get_X_y_containers(output_width, output_height)

        while X['start_img'].shape[0] < batch_size:
            try:
                img_row = img_metadata.sample(1)

                start_img0 = cv2.imread(raw_image_dir + img_row.filename_start.values[0])
                end_img0 = cv2.imread(raw_image_dir + img_row.filename_end.values[0])


                start_img_coords = Coords(0, 0, start_img0.shape[1], start_img0.shape[0])
                end_img_coords = Coords(0, 0, end_img0.shape[1], end_img0.shape[0])
                start_box_coords = Coords(img_row.x0_start, img_row.y0_start,
                                          img_row.x1_start, img_row.y1_start)

                end_box_coords = Coords(img_row.x0_end, img_row.y0_end,
                                        img_row.x1_end, img_row.y1_end)
                start_img, start_box = crop_and_resize(start_img0, start_img_coords,
                                                       start_box_coords, output_width,
                                                       output_height, random_crop=False)
                start_img, start_box = set_array_dims(start_img, start_box)
                start_img, end_img = (rough_normalization(img) for img in (start_img, end_img))
                for image_crop in range(crops_per_image):
                    end_img, end_box = crop_and_resize(end_img0, end_img_coords,
                                                       end_box_coords, output_width,
                                                       output_height, random_crop=True)
                    end_img, end_box = set_array_dims(end_img, end_box)
                    X['start_img'] = np.concatenate([X['start_img'], start_img])
                    X['start_box'] = np.concatenate([X['start_box'], start_box])
                    X['end_img'] = np.concatenate([X['end_img'], end_img])
                    y['end_box'] = np.concatenate([y['end_box'], end_box])

                # Limit output array sizes if larger than batch
                for field in X:
                    X[field] = X[field][:batch_size]
                y['end_box'] = y['end_box'][:batch_size]
            except:
                pass
            yield X, y
        # yield [X['start_img'], X['start_box'], X['end_img']], y

if __name__ == '__main__':
    my_gen = master_generator(crops_per_image=2, batch_size=50)
    X, y = next(my_gen)
    for i in range(10):
        show_img(X['start_img'][i], X['start_box'][i])
        show_img(X['end_img'][i], y['end_box'][i])
