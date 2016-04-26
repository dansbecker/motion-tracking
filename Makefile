work/.folder_structure_sentinel:

	mkdir -p data/ILSVRC/bb
	mkdir data/ILSVRC/images

	mkdir -p data/alov/bb/bb
	mkdir data/alov/frames

	mkdir -p work/ILSVRC
	mkdir work/alov

	touch work/.data_folder_structure_sentinel

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
work/.imagenet_training_bbox_sentinel:
	curl http://image-net.org/image/ilsvrc2014/ILSVRC2014_DET_bbox_train.tgz \
		-o data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz
	tar xvf data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz
	mv ILSVRC2014_DET_bbox_train/ data/ILSVRC/bb/training
	rm data/ILSVRC/bb/ILSVRC2014_DET_bbox_train.tgz
	touch work/.imagenet_training_bbox_sentinel

# Validation Bounding Boxes #
work/.imagenet_validation_bbox_sentinel:
	curl http://image-net.org/image/ilsvrc2013/ILSVRC2013_DET_bbox_val.tgz \
		-o data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz
	tar xvf data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz
  rm data/ILSVRC/bb/ILSVRC2013_DET_bbox_val.tgz
	mv ILSVRC2013_DET_bbox_val/ data/ILSVRC/bb/validation
  touch work/.imagenet_validation_bbox_sentinel

work/ILSVRC/parsed_training_bb.csv: work/.imagenet_training_bbox_sentinel
	bash code/data_setup/parse_xml.sh data/ILSVRC/bb/training/ \
		work/ILSVRC/parsed_training_bb.csv

work/ILSVRC/parsed_validation_bb.csv: work/.imagenet_validation_bbox_sentinel
	bash code/data_setup/parse_xml.sh data/ILSVRC/bb/validation/ \
				work/ILSVRC/parsed_validation_bb.csv


data/fall11_imagenet_urls.txt:
	curl http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz \
	  -o data/imagenet_fall11_urls.tgz
	tar xvf data/imagenet_fall11_urls.tgz
	rm data/imagenet_fall11_urls.tgz
	mv fall11_urls.txt data/fall11_imagenet_urls.txt


work/.imagenet_training_images_sentinel: code/data_setup/download_images.py data/fall11_imagenet_urls.txt
	python code/data_setup/download_images.py "training"
	touch work/.imagenet_training_images_sentinel

work/.imagenet_validation_images_sentinel: code/data_setup/download_images.py data/fall11_imagenet_urls.txt
	python code/data_setup/download_images.py "validation"
	touch work/.imagenet_validation_images_sentinel


imagenet_bb: work/ILSVRC/parsed_training_bb.csv work/ILSVRC/parsed_validation_bb.csv
data: work/.folder_structure_sentinel ILSVRC
imagenet_images: work/.imagenet_training_images_sentinel work/.imagenet_validation_images_sentinel
