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
        filename: str

    Output:
    ------
        parsed_df: pandas DataFrame
    """

    parsed_df = pd.read_table(bbox_dir + filename, sep=' ', header=None)
    parsed_df['filename'] = filename
    parsed_df['directory'] = (parsed_df['filename']
                                .apply(lambda fn: fn.split('_')[0]))

    return parsed_df


def add_filepaths(parsed_df, input_dir, output_dir):
    """Add the necessary filepaths for saving and later accessing.

    Args:
    ----
        parsed_df: pandas DataFrame
        input_dir: str
        output_dir: str

    Returns:
    -------
        parsed_df: pandas DataFrame
    """

    # The video numbers in the filenames are zero padded on the left.
    parsed_df['vid_num'] = (parsed_df['frame'].astype(str)
                               .apply(lambda frame: frame.zfill(8)))
    parsed_df['filename'] = (parsed_df['filename']
                                .apply(lambda filename:
                                    filename.replace('.ann', '')))

    # Need to replace the filename ext. in the DataFrame with `.jpg` and provide
    # the full relative input and output paths.
    parsed_df['input_filepath'] = (input_dir + parsed_df['directory_path'] +
        '/' + parsed_df['filename'] + '/' + parsed_df['vid_num'] + '.jpg')
    parsed_df['output_filepath'] = (output_dir + parsed_df['filename'] +
        '_' + parsed_df['vid_num'] + '.jpg')
    # To be used as the full path in the image generator.
    parsed_df['jpg_filename'] = (parsed_df['output_filepath']
                                    .apply(lambda filepath:
                                        filepath.split('/')[3]))

    return parsed_df


def cp_files(parsed_df, input_dir, output_dir):
    """Copy over the files given by the attributes in the dataframe.

    Args:
    -----
        parsed_df: pandas DataFrame
        input_dir: str
        output_dir: str

    Returns:
    -------
        filepaths_df: pandas DataFrame
    """

    filepaths_df = add_filepaths(parsed_df, input_dir, output_dir)
    for input_fp, output_fp in zip(filepaths_df['input_filepath'],
                                   filepaths_df['output_filepath']):
        shutil.copy(input_fp, output_fp)

    return filepaths_df


def fix_box_coords(filepaths_df):
    """Associate the correct bounding box coords with the right variables.

    Args:
    ----
        filepaths_df: pandas DataFrame

    Returns:
    -------
        filepaths_df: pandas DataFrame
    """

    # Rename to match the project terminology and what's used with imagenet.
    Xs = filepaths_df[['x1', 'x2', 'x3', 'x4']]
    filepaths_df['x_max'] = np.max(Xs, axis=1)
    filepaths_df['x_min'] = np.min(Xs, axis=1)

    Ys = filepaths_df[['y1', 'y2', 'y3', 'y4']]
    filepaths_df['y_max'] = np.max(Ys, axis=1)
    filepaths_df['y_min'] = np.min(Ys, axis=1)

    filepaths_df.drop(['x1', 'x2', 'x3', 'x4', 'y1', 'y2', 'y3', 'y4'],
            axis=1, inplace=True)
    filepaths_df.rename(columns={'x_min': 'x0', 'x_max': 'x1',
            'y_min': 'y0', 'y_max': 'y1'}, inplace=True)

    return filepaths_df


def calc_frame_pairs(frames_df):
    """Add columns denoting the relevant info for the current and previous frame.

    Each video has loads of frames, and we need to get information for subsequent
    frames in the same row in the DataFrame. This will make it easy for our image
    generator to easily cycle through pairs. We'll accomplish this by taking the
    `frames_df`, lopping off the last row, placing in a filler row, and merging it
    back onto the original `frames_df`. The rest will be cleanup.

    Args:
    ----
        frames_df: pandas DataFrame

    Returns:
    -------
        save_df: pandas DataFrame
    """

    filler_row = pd.DataFrame(np.zeros((1, frames_df.shape[1])),
                              columns=frames_df.columns)
    less_one_df = frames_df[:-1]
    lagged_df = pd.concat([filler_row, less_one_df], axis=0)

    lagged_cols = [col + '_start' for col in frames_df.columns]
    lagged_df.columns = lagged_cols
    lagged_df.reset_index(inplace=True, drop=True)

    end_cols = [col + '_end' for col in frames_df.columns]
    frames_df.columns = end_cols
    merged_df = pd.concat([lagged_df, frames_df], axis=1)

    max_frames_df = merged_df.groupby('filename_start')['frame_start'].max()
    max_frames_df.name = 'max_frame'
    temp_df = merged_df.join(max_frames_df, on='filename_start')
    save_df = temp_df.query('max_frame != frame_start')

    return save_df

if __name__ == '__main__':
    input_dir = sys.argv[1]  # Holds location to find alov `.ann` files.
    output_filepath = sys.argv[2]  # File to give to the resulting .csv.
    output_dir = sys.argv[3]  # Output directory to save resulting .csv in.

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
    filepaths_df = cp_files(parsed_df, frames_dir, output_dir)

    fixed_coords_df = fix_box_coords(filepaths_df)
    save_df = calc_frame_pairs(fixed_coords_df)
    # Only keep what we'll need for the generator.
    keep_cols = ['x0_start', 'y0_start', 'x1_start', 'y1_start',
                 'jpg_filename_start', 'x0_end', 'x1_end', 'y0_end', 'y1_end',
                 'jpg_filename_end']
    save_df = save_df[keep_cols]
    save_df.rename(columns={'jpg_filename_start': 'filename_start',
                            'jpg_filename_end': 'filename_end'}, inplace=True)
    save_df.to_csv(output_filepath, index=False)
