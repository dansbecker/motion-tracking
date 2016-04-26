"""A script for parsing the Imagenet bounding box XML files."""

import sys
import os
import itertools
import xml.etree.ElementTree as ET 

def parse_file(bbox_dir, filepath): 
    tree = ET.parse(bbox_dir + filepath)
    root = tree.getroot()

    output_lst = []
    size_node = root.find('size')
    width = size_node.find('width').text
    height = size_node.find('height').text

    for child in root.findall('./'): 
        if child.tag == "object": 
            object_lst = []
            object_lst.append(width)
            object_lst.append(height)
            name = child.find('name').text
            subcategory = child.find('subcategory')
            subcategory = None if subcategory is None else subcategory.text

            bounding_box = child.find('bndbox')
            xmin = bounding_box.find('xmin').text
            xmax = bounding_box.find('xmax').text
            ymin = bounding_box.find('ymin').text
            ymax = bounding_box.find('ymax').text

            object_lst.append(name)
            object_lst.append(subcategory)
            object_lst.append(xmin)
            object_lst.append(xmax)
            object_lst.append(ymin)
            object_lst.append(ymax)
            output_lst.append(object_lst)

    return output_lst
    
if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    xml_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_xml_filenames = itertools.chain(*xml_files_by_dir)

    end_lst = []
    for filename in bbox_xml_filenames: 
        end_lst.extend(parse_file(bbox_dir, filename))
    # parsed_results = [parse_file(bbox_dir, filename) for \
    #      filename in bbox_xml_filenames]


