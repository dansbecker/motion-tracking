"""A script for parsing the Alov bounding box `.ann` files."""

import sys
import os
import itertools
import pandas as pd
import xml.etree.ElementTree as ET 


if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    ann_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_ann_filenames = itertools.chain(*ann_files_by_dir)

    end_lst = []
    for filename in bbox_xml_filenames: 
        end_lst.extend(parse_file(bbox_dir, filename))

