"""A script for parsing the Alov bounding box `.ann` files."""

import sys
import os
import itertools
import shutil
import pandas as pd
import numpy as np

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

    # Need to replace the filename ext. in the DataFrame with `.jpg` and provide 
    # the full relative path point to what to move. 
    parsed_df['input_filepath'] = (input_dir +   
        parsed_df['directory_path'] + '/' + 
        parsed_df['filename'] + '/' + parsed_df['vid_num'] 
        + '.jpg')
    # Need to provide the full relative output path to point to where to move. 
    # The `filename` is being used as part of what to save so that images can be
    # identified (there are lot with the same number, so we need an identifying
    # feature). 
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
    """ 

    for input_fp, output_fp in zip(filepaths_df['input_filepath'], 
            filepaths_df['output_filepath']): 
        shutil.copy(input_fp, output_fp)

def calc_frame_pairs(end_df): 
    """Add columns denoting the relevant info for the current and previous frame. 

    Each video has loads of frames, where every 5th (starting with the 1st) has 
    a bounding box. We need to get information for subsequent frames in the same
    row in the DataFrame, so it's easy for our generator to easily cycle through 
    pairs. Here, we'll clean up the DataFrame and make sure that we end up with 
    columns that distinguish between the current and previous frames filepaths as
    well as bounding boxes. 
    
    Args: 
    ----
        start_df: pandas DataFrame

    Returns: 
    -------
        save_df: pandas DataFrame
    """

    filler_row = pd.DataFrame(np.zeros((1, end_df.shape[1])), columns=end_df.columns)
    less_one_df = end_df[:-1]
    lagged_df = pd.concat([filler_row, less_one_df], axis=0)

    lagged_cols = [col + '_start' for col in end_df.columns]
    lagged_df.columns = lagged_cols
    lagged_df.reset_index(inplace=True, drop=True)

    end_cols = [col + '_end' for col in end_df.columns]
    end_df.columns = end_cols
    merged_df = pd.concat([lagged_df, end_df], axis=1)
    
    max_frames_df = merged_df.groupby('filename_start')['frame_start'].max()
    max_frames_df.name = 'max_frame'
    temp_df = merged_df.join(max_frames_df, on='filename_start')
    save_df = temp_df.query('max_frame != frame_start')

    return save_df
    
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
    parsed_df.reset_index(inplace=True, drop=True)

    frames_dir = input_dir + 'frames/'
    filepaths_df = add_filepaths(parsed_df, frames_dir, output_dir)
    cp_files(filepaths_df)

    # Rename to match the project terminology and what's used with imagenet. 
    filepaths_df['x_max'] = np.max(filepaths_df[['x1', 'x2', 'x3', 'x4']], axis=1)
    filepaths_df['x_min'] = np.min(filepaths_df[['x1', 'x2', 'x3', 'x4']], axis=1)
    filepaths_df['y_max'] = np.max(filepaths_df[['y1', 'y2', 'y3', 'y4']], axis=1)
    filepaths_df['y_min'] = np.min(filepaths_df[['y1', 'y2', 'y3', 'y4']], axis=1)
    filepaths_df.drop(['x1', 'x2', 'x3', 'x4', 'y1','y2', 'y3', 'y4'], axis=1, inplace=True)
    filepaths_df.rename(columns={'x_min':'x0', 'x_max':'x1', 'y_min':'y0', 'y_max':'y1'}, inplace=True)
    save_df = calc_frame_pairs(filepaths_df)
    # Only keep what we'll need for the generator. 
    keep_cols = ['x0_start', 'y0_start', 'x1_start', 'y1_start', 
            'jpg_filename_start', 'x0_end', 'x1_end', 'y0_end', 'y1_end', 
            'jpg_filename_end']
    save_df = save_df[keep_cols]

    save_df.rename(columns={'jpg_filename_start': 'filename_start', 
                            'jpg_filename_end': 'filename_end'}, inplace=True)
    save_df.to_csv(output_filepath, index=False)
