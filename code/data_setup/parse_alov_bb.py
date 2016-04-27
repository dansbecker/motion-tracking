"""A script for parsing the Alov bounding box `.ann` files."""

import sys
import os
import itertools

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

    with open(bbox_dir + filename) as f: 
        output_lst = [line.split() for line in f]

    return output_lst 

if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    ann_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_ann_filenames = itertools.chain(*ann_files_by_dir)

    first = bbox_ann_filenames.next()
    out = parse_file(bbox_dir, first)
    #end_lst = []
    # for filename in bbox_xml_filenames: 
    #    end_lst.extend(parse_file(bbox_dir, filename))

