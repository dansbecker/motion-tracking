"""A script for parsing the Alov bounding box `.ann` files."""

import sys
import os
import itertools
import pandas as pd

def parse_file(bbox_dir, filename): 
    """Parse an individual `.ann` file and output the relevant elements.

    Args: 
    ----
        bbox_dir: str
            Holds the directory path to the `.ann` file. 
        filename: str
            Holds the name of the `.ann` file. 

    Output: 
    ------
        output_lst: list of lists 
            Holds the elements grabbed from the `.ann` file. 
    """

    output_lst = []

    with open(bbox_dir + filename) as f: 
        output_lst = [parse_line(filename, line) for line in f]

    return output_lst 

def parse_line(filename, line): 
    """Parse an individual line from an `.ann` file. 

    Args: 
    ----
        filename: str
        line: str

    Output: 
    ------
        object_lst: list
            Parsed list of elements. 
    """

    filename_parts = filename.split('_')
    line_parts = line.split()
    object_lst = [filename_parts[0], filename, line_parts[0], 
            line_parts[1], line_parts[2], line_parts[3], 
            line_parts[4], line_parts[5], line_parts[6], 
            line_parts[7], line_parts[8]]

    return object_lst
    

if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    ann_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_ann_filenames = itertools.chain(*ann_files_by_dir)

    all_bboxes = (parse_file(bbox_dir, filename) for filename in bbox_ann_filenames)
    end_lst = list(itertools.chain(*all_bboxes))


    cols = ['directory_path', 'filename', 'frame', 'x1', 'y1', \
            'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    output_df = pd.DataFrame(data=end_lst, columns=cols)
    output_df.to_csv(output_filepath, index=False)
