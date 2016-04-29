"""A script for parsing the Alov bounding box `.ann` files."""

import sys
import os
import itertools
import shutil
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
        parsed_df: pandas DataFrame 
            Holds the elements grabbed from the `.ann` file. 
    """

    parsed_df = pd.read_table(bbox_dir + filename, sep=' ', header=None)
    parsed_df['filename'] = filename
    parsed_df['directory'] = parsed_df['filename'].apply(lambda fn: fn.split('_')[0])

    return parsed_df

def add_filepaths(parsed_df, input_dir, output_dir): 
    """Add the necessary filepaths for saving and later accessing. 

    Args: 
    ----
        parsed_df: pandas DataFrame
            Holds info from parsed alov annotation files. 
        output_dir: str
            Holds the directory of where to copy the file. 
    """
    
    # The video numbers are zero padded on the left. 
    parsed_df['vid_num'] = parsed_df['frame'].astype(str).apply(lambda 
            frame: frame.zfill(8))
    parsed_df['filename'] = (parsed_df['filename']
            .apply(lambda filename: filename.replace('.ann', '')))

    # Need to replace the filename ext. in the DataFrame with `.jpg` and 
    # provide the full relative path point to what to move. 
    parsed_df['input_filepath'] = (input_dir +   
        parsed_df['directory_path'] + '/' + 
        parsed_df['filename'] + '/' + parsed_df['vid_num'] 
        + '.jpg')
    # Need to provide the full relative output path to point to 
    # where to move. The `filename` is being used as part of 
    # what to save so that images can be identified (there are 
    # lot with the same number, so we need an identifying feature). 
    parsed_df['output_filepath'] = (output_dir +
        parsed_df['filename'] + '_' + parsed_df['vid_num'] 
        + '.jpg')
    
    parsed_df['jpg_filename'] = (parsed_df['output_filepath']
            .apply(lambda filepath: filepath.split('/')[3]))

    return parsed_df

def cp_files(filepaths_df): 
    """Copy over the files given by the attributes in the dataframe. 

    Args: 
    -----
        filepaths_df: pandas DataFrame
            Holds 3 columns, each of which holds a piece that 
            determines the input filepaths. We'll copy the files 
            at these filepaths to the output_dir.
    """
    
    for input_fp, output_fp in zip(filepaths_df['input_filepath'], 
            filepaths_df['output_filepath']): 
        shutil.copy(input_fp, output_fp)

def add_framepaths(save_df): 
    """Add columns into the dataframe pointing to pairs of frames. 

    For the image generator, we'll need to easily grab successive
    frames, and have the bounding boxes for these frames readily 
    available. To do this, we'll create columns for the filenames
    for each frame, as well as columns for their bounding boxes. 
    They will be differentiated by prefixes of current and previous. 

    Args: 
    ----
        save_df: pandas DataFrame
    """

    save_df = calc_frame_pairs(save_df)

def calc_frame_pairs(save_df): 
    """Add columns denoting the filepaths of current and previous frames. 

    Each video has loads of frames, where every 5th (starting with 
    the 1st) has a bounding box. For every frame except the first 
    and the last, we can simply add 5 to it. For the first, 
    we need it to always be the previous frame since there's nothing
    before it. For the last, we need it to always be the current, since 
    there is nothing after it. 
    """
    pass


if __name__ == '__main__': 
    input_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    output_dir = sys.argv[3]

    bbox_dir = input_dir + 'bb/'
    ann_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_ann_filenames = itertools.chain(*ann_files_by_dir)


    cols = ['frame', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 
            'y4', 'filename', 'directory_path']
    parsed_df = pd.concat(parse_file(bbox_dir, filename) for 
            filename in bbox_ann_filenames)
    parsed_df.columns = cols

    frames_dir = input_dir + 'frames/'
    filepaths_df = add_filepaths(parsed_df, frames_dir, output_dir)
    cp_files(filepaths_df)

    # Now that the copying is done, only a subset of the columns 
    # are necessary. 
    keep_cols = ['frame', 'x1', 'y1', 'x3', 'y3', 'jpg_filename', 
            'filename']
    save_df = filepaths_df[keep_cols]
    '''
    save_df = add_framepaths(save_df)
    save_df.to_csv(output_filepath, index=False)
    '''
