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
    
def cp_files(filepaths_df, input_dir, output_dir): 
    """Move the files given by the attributes in the input array. 

    Args: 
    -----
        filepaths_df: pandas DataFrame
            Holds 3 columns, each of which holds a piece that 
            determines the input filepaths. We'll copy the files 
            at these filepaths to the output_dir.
        input_dir: str
            Holds the first part of the input directory path. 
        output_dir: str
            Holds the directory of where to copy the file. 
    """
    
    filepaths_df['vid_num'] = filepaths_df['frame'].astype(str).apply(lambda 
            frame: frame.zfill(8))
    filepaths_df['filename'] = filepaths_df['filename'].apply(lambda filename: 
            filename.replace('.ann', ''))
    filepaths_df['input_filepath'] = input_dir + \
            filepaths_df['directory_path'] + '/' + filepaths_df['filename'] + \
            '/' + filepaths_df['vid_num'] + '.jpg'
    filepaths_df['output_filepath'] = output_dir + filepaths_df['filename'] + \
            '_' + filepaths_df['vid_num'] + '.jpg'
    for input_fp, output_fp in zip(filepath_df['input_filepath'], 
            filepath_df['output_filepath']): 
        cp_file(input_fp, output_fp)

def cp_file(input_filepath, output_dir): 
    """Copy an individual file to the output_dir. 

    Args: 
    ----
        input_filepath: str 
        output_dir: str
    """
     
    cp_command = "cp {in_fp} {out_dir}".format(in_fp=input_filepath, 
            out_dir=output_dir)
    os.system(cp_command)

if __name__ == '__main__': 
    input_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    output_dir = sys.argv[3]

    bbox_dir = input_dir + 'bb/'
    ann_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_ann_filenames = itertools.chain(*ann_files_by_dir)
    all_bboxes = (parse_file(bbox_dir, filename) for filename in bbox_ann_filenames)
    end_lst = list(itertools.chain(*all_bboxes))

    cols = ['directory_path', 'filename', 'frame', 'x1', 'y1', \
            'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    output_df = pd.DataFrame(data=end_lst, columns=cols)
    output_df.to_csv(output_filepath, index=False)

    frames_dir = input_dir + 'frames/'
    filepath_cols = ['directory_path', 'filename', 'frame']
    filepath_df = output_df[filepath_cols]
    cp_files(filepath_df, frames_dir, output_dir)
