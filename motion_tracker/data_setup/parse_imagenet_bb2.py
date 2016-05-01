"""A short script for futher filtering the imagenet bounding boxes.

The original imagenet bounding boxes csv contains bounding boxes for images that
failed during download. This script will filter out those bounding boxes so that 
the bounding boxes csv is as lightweight as possible. 
"""

import sys
import os
import pandas as pd

def parse_imagenet_bb(raw_image_dir, parsed_bb_path):
    '''Return df with data about images and bounding boxes.

    Data captured corresponds to xml files obtained from ImageNet.
    '''

    # paper says they use .66, but that creates search areas 
    # larger than image. So, 0.5 may be more appropriate.
    max_box_frac_of_width =  0.66
    max_box_frac_of_height = 0.66
    images_successfully_downloaded = set(os.listdir(raw_image_dir))
    bbox_df = (pd.read_csv(parsed_bb_path)
                    .assign(box_height = lambda df: df.y1 - df.y0,
                            box_width = lambda df: df.x1 - df.x0)
                    .assign(box_frac_of_height = lambda df: 
                        df.box_height / df.height,
                            box_frac_of_width = lambda df:  
                        df.box_width / df.width)
                    .query('filename in @images_successfully_downloaded')
                    .query('box_frac_of_height < @max_box_frac_of_height')
                    .query('box_frac_of_width < @max_box_frac_of_width'))

    # Assuming the path ends in .csv, just insert a 2 in front 
    # of the .csv extension. 
    out_filepath = parsed_bb_path[:-4] + '2' + '.csv'
    bbox_df.to_csv(out_filepath, index=False)

if __name__ == '__main__': 
    raw_image_dir = sys.argv[1]
    parsed_bb_path = sys.argv[2]

    parse_imagenet_bb(raw_image_dir, parsed_bb_path)
