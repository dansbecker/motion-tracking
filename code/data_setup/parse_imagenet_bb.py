"""A script for parsing the Imagenet bounding box XML files."""

import sys
import os
import itertools
import xml.etree.ElementTree as ET 

def parse_file(bbox_dir, filepath): 
    tree = ET.parse(bbox_dir + filepath)
    print root.findall('./')
    pass

if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    xml_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_xml_filenames = itertools.chain(*xml_files_by_dir)

    parsed_results = [parse_file(bbox_dir, filename) for \
            filename in bbox_xml_filenames]


