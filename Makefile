.folder_structure_sentinel: 

	mkdir -p data/ILSVRC/bb
	mkdir data/ILSVRC/images

	mkdir -p data/alov/bb/bb
	mkdir data/alov/frames

	mkdir -p work/ILSVRC
	mkdir work/alov

	touch .data_folder_structure_sentinel

####################
# ALOV300++ Datset # 
####################

# Frames # 
data/alov/frames/alov300++_frames.zip: 
	curl http://isis-data.science.uva.nl/alov/alov300++_frames.zip \
		-o data/alov/frames/alov300++_frames.zip
	
	tar xvf data/alov/frames/alov300++_frames.zip 

	mv imagedata++ data/alov/frames/

# Bounding Boxes # 
data/alov/bb/alov300++GT_txtFiles.zip: 
	curl http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip \
		-o data/alov/bb/alov300++GT_txtFiles.zip

	unzip data/alov/bb/alov300++GT_txtFiles.zip 

	mv alov300++_rectangleAnnotation_full/ data/alov/bb/bb

alov: data/alov/frames/alov300++_frames.zip \
	data/alov/bb/alov300++GT_txtFiles.zip 

##############
# ILSVRC2014 # 
##############

# Training Bounding Boxes # 
data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz: 
	curl http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_bbox_train.tgz \
		-o data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz

	tar xvf data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz

	mv ILSVRC2014_DET_bbox_train/ data/ILSVRC/bb/training

# Validation Bounding Boxes # 
data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz: 
	curl http://image-net.org/image/ilsvrc2013/ILSVRC2013_DET_bbox_val.tgz \
		-o data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz

	tar xvf data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz

	mv ILSVRC2013_DET_bbox_val/ data/ILSVRC/bb/validation
	
work/ILSVRC/parsed_training_bb.csv: 
	bash code/data_setup/parse_xml.sh data/ILSVRC/bb/training/ \
		work/ILSVRC/parsed_training_bb.csv

work/ILSVRC/parsed_validation_bb.csv: 
	bash code/data_setup/parse_xml.sh data/ILSVRC/bb/validation/ \
				work/ILSVRC/parsed_validation_bb.csv

ILSVRC: data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz \
	data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz \
	work/ILSVRC/parsed_training_bb.csv \
	work/ILSVRC/parsed_validation_bb.csv 

data: .folder_structure_sentinel ILSVRC 
