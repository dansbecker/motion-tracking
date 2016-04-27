"""A script for parsing the Imagenet bounding box XML files."""

import sys
import os
import itertools
import pandas as pd
import xml.etree.ElementTree as ET 

def parse_file(bbox_dir, filename): 
    """Parse an individual XML file and grab the relevant elements. 

    Args: 
    ----
        bbox_dir: str
            Holds the directory path to the XML file. 
        filename: str
            Holds the name of the XML file. To be attached to the 
            `bbox_dir` to be read in and parsed. 

    Output: 
    ------
        output_lst: list of lists 
            Holds lists of the values parsed from an XML file. 
    """
    tree = ET.parse(bbox_dir + filename)
    root = tree.getroot()

    output_lst = []
    width, height = parse_size(root)

    for child in root.findall('./'): 
        if child.tag == "object": 

            name = child.find('name').text
            subcategory = child.find('subcategory')
            subcategory = None if subcategory is None else subcategory.text

            bounding_box = child.find('bndbox')
            xmin, xmax, ymin, ymax = parse_bb(bounding_box)

            object_lst = [width, height, name, subcategory, xmin, 
                    xmax, ymin, ymax]

            output_lst.append(object_lst)

    return output_lst

def parse_size(root): 
    """Parse the width and height of the image out of the root node. 

    Args: 
    ----
        root: xml.etree.ElementTree.Element

    Output: 
    ------
        width: str
        height str
    """

    size_node = root.find('size')
    width = size_node.find('width').text
    height = size_node.find('height').text

    return width, height

def parse_bb(bounding_box): 
    """Parse the coordinates of the bounding box from the bounding box node.

    Args: 
    ----
        bounding_box: xml.etree.ElementTree.Element

    Output: 
    ------
        xmin: str
        xmax: str
        ymin: str
        ymax: str
    """
    
    xmin = bounding_box.find('xmin').text
    xmax = bounding_box.find('xmax').text
    ymin = bounding_box.find('ymin').text
    ymax = bounding_box.find('ymax').text

    return xmin, xmax, ymin, ymax
    
if __name__ == '__main__': 
    bbox_dir = sys.argv[1]
    output_filepath = sys.argv[2]
    xml_files_by_dir = (i[2] for i in os.walk(bbox_dir))
    bbox_xml_filenames = itertools.chain(*xml_files_by_dir)

    all_bboxes = (parse_file(bbox_dir, filename) for filename in bbox_xml_filenames)
    end_lst = list(itertools.chain(*all_bboxes))

    cols = ['width', 'height', 'name', 'subcategory', 'xmin', 'xmax', 
        'ymin', 'ymax']
    output_df = pd.DataFrame(data=end_lst, columns=cols)
    output_df.to_csv(output_filepath, index=False)
