"""A script for building image generators for net modeling."""

from motion_tracker.utils.image_cropping import Coords, crop_and_resize

def master_generator(num_video_crops, num_image_crops, batch_size): 
    """Generator yielding a batch of images. 

    Args: 
    ----
        num_video_crops: int
            Number of random crops of the current frame to take. 
            Denoted as k3 in the paper, with a value of 10. 
        num_image_crops: int 
            Number of random crops of the randomly sampled image 
            to take. Dentoed as k4 in the paper, with a value of 10. 
        batch_size: int
            Number of items in list to be returned. 
    """
    # To switch witch training images we draw from first in 
    # a batch. 
    imagnet = True
    imagenet_gen = imagenet_generator(num_image_crops)

    while True: 
        
        out_imgs = []
        out_boxes = []
        video_pairs = 0
        images = 0
        # Might need to add in some check or something in the 
        # case that the num_video_crops and num_image_crops can't
        # logically (and in an alternating way) add up to the 
        # batch size. This while loop figures out how many pairs 
        # of video frames to grab and how many images to grab. 
        while num_obs < batch_size: 
            if imagenet: 
                images += 1
                num_obs += num_video_crops
            else: 
                video_pairs += 1
                num_obs += num_image_crops
            imagenet = not imagenet
        
        # Now grab the actual images and video based off the number
        # of pairs and sets of random croppings we need. 
        for idx in range(images): 
            imgs, boxes = next(imagenet_gen)
            out_imgs.append(imgs)
            out_boxes.append(boxes)
        for idx in range(video_pairs): 
            imgs, boxes = next(alov_gen)
            out_imgs.append(imgs)
            out_boxes.append(boxes)

        yield out_imgs, out_boxes
            
def imagenet_generator(num_image_crops):
    """Generator yielding images and bounding boxes. 

    Args: 
    ----
        num_image_crops: int
            Number of image crops to take of each randomly drawn 
            image. 

    Outputs:
    -------
        imgs_lst: list of images
        boxes_lst: list of bounding box coords
    """

    img_metadata = pd.read_csv('work/imagenet/parsed_bb2.csv')
    output_width = 256
    output_height = 256

    while True:
        imgs_lst = []
        boxes_lst = []
        batch_df = img_metadata.sample(1)

        for _, row in batch_df.iterrows():
            for image_crop in range(num_image_crops): 
                try:
                    img = cv2.imread(raw_image_dir + row.filename)
                    img_coords = Coords(0, 0, row.width, row.height)
                    box_coords = Coords(row.xmin, row.ymin, 
                            row.xmax, row.ymax)
                    img, box = crop_and_resize(img, 
                            img_coords, box_coords, output_width,
                            output_height, random_crop=True)

                    imgs_lst.append(img)
                    box_lst.append(box)
                except:
                    print('Failed on ', row.filename)
    
            yield imgs_lst, box_lst
