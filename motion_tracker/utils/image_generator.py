import pandas as pd
import numpy as np
import cv2
from motion_tracker.utils.image_cropping import Coords, crop_and_resize
from motion_tracker.utils.crop_vis import show_img


class ImagePairAndBoxGen(object):
    def __init__(self, crops_per_image = 10, batch_size=50,
                 output_width = 256, output_height = 256,
                 desired_dim_ordering='th'):
        """Generator yielding dictionary of image and bounding boxes.
        This is an abstract class and will not be called directly.

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
            desired_dim_ordering: Either 'th' or 'tf' to determine where channel index is
        """
        self.crops_per_image = crops_per_image
        self.batch_size = batch_size
        self.output_width = output_width
        self.output_height = output_height
        self.desired_dim_ordering = desired_dim_ordering

    @property
    def accumulators_full(self):
        return self.X['start_img'].shape[0] >= self.batch_size

    def flow(self):
        """The function yielding an iterator."""
        while True:
            self.reset_accumulators()
            while not self.accumulators_full:
                self.add_to_accumulators()
            self.remove_accumulator_overflow()
            self.ensure_dim_ordering()
            yield self.X, self.y


    def reset_accumulators(self):
        """Empties all contents from previous batch. Sets up for next batch"""
        empty_img_accumulator = np.zeros([0, self.output_width, self.output_height, 3]).astype('uint8')
        empty_img_mask_accumulator = np.zeros([0, self.output_width, self.output_height, 1]).astype('uint8')
        empty_box_accumulator = np.zeros([0, 4]).astype('uint16')
        empty_digit_accumulator = np.zeros([0]).astype('uint16')
        self.X = {'start_img': empty_img_accumulator,
                  'start_box': empty_box_accumulator,
                  'start_box_mask': empty_img_mask_accumulator,
                  'end_img': empty_img_accumulator}
        self.y = {'x0': empty_digit_accumulator,
                  'y0': empty_digit_accumulator,
                  'x1': empty_digit_accumulator,
                  'y1': empty_digit_accumulator}

    def remove_accumulator_overflow(self):
        """Cuts extra data if we have more data than batch size"""
        for field in self.X:
            self.X[field] = self.X[field][:self.batch_size]
        for field in self.y:
            self.y[field] = self.y[field][:self.batch_size]

    def ensure_dim_ordering(self):
        if self.desired_dim_ordering == 'th': # swap because data comes in 'tf' format
            imgs_to_flip  = ['start_img', 'end_img', 'start_box_mask']
            for img in imgs_to_flip:
                self.X[img] = self.X[img].swapaxes(1,3)

    def coords_to_mask(self, start_box):
        # add a 1 as 1st dim to allow concatenating boxes from many images into 1 array
        # add a 1 as last dim for consistency with image dim count
        output = np.zeros([1, self.output_height, self.output_width, 1])
        output[start_box.y0:start_box.y1, start_box.x0:start_box.x1] = 1
        return output

    def add_to_accumulators(self):
        have_good_img = False
        while not have_good_img:
            img_row = self.img_metadata.sample(1)
            raw_start_img, raw_start_box, raw_end_img, raw_end_box, have_good_img = self.read_raw_imgs_and_boxes(img_row)

        start_img_coords = Coords(0, 0, raw_start_img.shape[1], raw_start_img.shape[0])
        end_img_coords = Coords(0, 0, raw_end_img.shape[1], raw_end_img.shape[0])

        start_img, start_box = crop_and_resize(raw_start_img, start_img_coords, raw_start_box,
                                               self.output_width, self.output_height,
                                               random_crop=False)
        start_box_mask = self.coords_to_mask(start_box)
        start_img, start_box = self.set_array_dims(start_img, start_box)
        for image_crop in range(self.crops_per_image):
            end_img, end_box = crop_and_resize(raw_end_img, end_img_coords, raw_end_box,
                                               self.output_width, self.output_height,
                                               random_crop=True)
            end_img, _ = self.set_array_dims(end_img, end_box)
            for name, data in (('start_img', start_img), ('start_box', start_box),
                               ('end_img', end_img), ('start_box_mask', start_box_mask)):
                self.X[name] = np.concatenate([self.X[name], data])
            for coord in self.y:
                self.y[coord] = np.concatenate([self.y[coord], end_box.as_dict()[coord]])


    def set_array_dims(self, img, box):
        ''' Adds an additional axis of length 1 to start of imgs and boxes.

            These dimensionality changes are necessary for merging to accumulators
            that have observations in the first axis
        '''

        out_img = img[np.newaxis].astype('uint8')
        out_box = box.as_array()[np.newaxis]
        return out_img, out_box



class CompositeGenerator(ImagePairAndBoxGen):
    """ Class to pull pairs of image/box pairs from multiple subgenerators
    (e.g. ImagenetGenerator and AlovGenerator)
    """
    def __init__(self, crops_per_image = 10, batch_size=50,
                 output_width = 256, output_height = 256,
                 desired_dim_ordering='th'):
        """See ImagePairAndBoxGen for argument definitions"""
        self.generators_to_draw_from =                                      \
                        [gen(batch_size=crops_per_image,
                                            crops_per_image=crops_per_image,
                                            output_width=output_width,
                                            output_height=output_height).flow()
                        for gen in (ImagenetGenerator, AlovGenerator)]

        super(CompositeGenerator, self).__init__(crops_per_image = crops_per_image,
                                               batch_size = batch_size,
                                               output_width = output_width,
                                               output_height = output_height,
                                               desired_dim_ordering = desired_dim_ordering)

    def add_to_accumulators(self):
        for img_source in self.generators_to_draw_from:
            subsource_X, subsource_y = next(img_source)
            for field in self.X:
                self.X[field] = np.concatenate([self.X[field], subsource_X[field]])
            for field in self.y:
                self.y[field] = np.concatenate([self.y[field], subsource_y[field]])

class ImagenetGenerator(ImagePairAndBoxGen):
    def __init__(self, crops_per_image=10, batch_size=50,
                 output_width = 256, output_height = 256,
                 desired_dim_ordering='tf'):
        """See ImagePairAndBoxGen for argument definitions"""
        self.raw_image_dir = 'data/imagenet/images/'
        self.img_metadata = pd.read_csv('work/imagenet/parsed_bb2.csv')
        super(ImagenetGenerator, self).__init__(crops_per_image = crops_per_image,
                                               batch_size = batch_size,
                                               output_width = output_width,
                                               output_height = output_height,
                                               desired_dim_ordering = desired_dim_ordering)


    def read_raw_imgs_and_boxes(self, img_row):
        have_good_image = False
        raw_img = cv2.imread(self.raw_image_dir + img_row.filename.values[0])
        raw_start_box = Coords(img_row.x0, img_row.y0, img_row.x1, img_row.y1)
        # verify image dimensions
        if (raw_img.shape[0] == img_row.height.iloc[0]) and \
           (raw_img.shape[1] == img_row.width.iloc[0]):
                have_good_image = True
        # reuse the one image/box for both start and end
        return raw_img, raw_start_box, raw_img, raw_start_box, have_good_image



class AlovGenerator(ImagePairAndBoxGen):
    def __init__(self, crops_per_image=10, batch_size=50,
                 output_width = 256, output_height = 256,
                 desired_dim_ordering='tf'):
        """See ImagePairAndBoxGen for argument definitions"""
        self.raw_image_dir = 'work/alov/images/'
        self.img_metadata = pd.read_csv('work/alov/parsed_bb.csv')
        super(AlovGenerator, self).__init__(crops_per_image = crops_per_image,
                                               batch_size = batch_size,
                                               output_width = output_width,
                                               output_height = output_height,
                                               desired_dim_ordering = desired_dim_ordering)

    def read_raw_imgs_and_boxes(self, img_row):
        have_good_image = True
        raw_start_img = cv2.imread(self.raw_image_dir + img_row.filename_start.values[0])
        raw_end_img = cv2.imread(self.raw_image_dir + img_row.filename_end.values[0])
        raw_start_box = Coords(img_row.x0_start, img_row.y0_start,
                                  img_row.x1_start, img_row.y1_start)
        raw_end_box = Coords(img_row.x0_end, img_row.y0_end,
                                img_row.x1_end, img_row.y1_end)
        return raw_start_img, raw_start_box, raw_end_img, raw_end_box, have_good_image

if __name__ == '__main__':
    img_size=256
    my_gen = CompositeGenerator(crops_per_image=2, batch_size=20,
                                output_height=img_size, output_width=img_size,
                                desired_dim_ordering='tf').flow()
    for i in range(10):
        show_img(X['start_img'][i],
                box={'start_box': X['start_box'][i]})
        show_img(X['end_img'][i],
                 np.array([y[coord][i] for coord in ('x0', 'y0', 'x1', 'y1')]))
