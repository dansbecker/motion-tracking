## The Data

The data used for this project are all publicly available. At this point, current data sources included the following: 

1. [ALOV300++ Dataset](http://www.alov300.org/)
2. [ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/)

### Data Structure

1. The ALOV300++ Dataset comes with frames from 314 videos, along with bounding boxes for select frames (typically every 5th frame). The raw video frames can be downloaded from the [Download Category](http://isis-data.science.uva.nl/alov/alov300++_frames.zip) link at the ALOV300++ Dataset link above, and the bounding boxes can be found at the [Download Ground Truth](http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip) link (they are also linked here). 

2. The ImageNet Large Scale Visual Recognition Challenge 2014 (ILSVRC2014) is an annual competition evaluating algorithms for object detection and image classification. The data gathered from this competition includes raw images used in the competition, as well as annotated bounding boxes of object classes in those raw images. The images can be downloaded via their URL's by following along with the instructions at the ImageNet [download-imageurls](http://image-net.org/download-imageurls) page, and the bounding boxes by following along with the instructions at the ImageNet [download-bboxes](http://image-net.org/download-bboxes) page.  

### Data Folder Structure

There are currently no folders with data showing in this folder. There are scripts meant to facilitate download - `mk_folders.sh` (located in this folder), the `Makefile` located in the main folder of this repository, and `parse_xml.sh` (located in the `code/data_setup` folder, accessible from the main folder of this repository).  

**Notes**: 

* `mk_folders.sh` - this will set up the folder structure to store the data, whereas the `Makefile` in the main folder will actually downloads the data and unzips it. The `parse_xml.sh` will parse the `xml` files that contain the bounding boxes for the ImageNet images (it will place the results in a `csv` called `parsed_ILSVRC_bb.csv`). 
