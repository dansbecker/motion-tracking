# Motion-Tracking

This project uses convolutional neural networks to track an object through sequential video frames.  It is
inspired by [Recent Advances in Offline Object Tracking](http://arxiv.org/pdf/1604.01802v1.pdf). We aimed to recreate and improve on these recent advancements.

Our work process included collecting data from the [ALOV300++](http://www.alov300.org/) and [ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/) datasets, augmenting the data through random croppings, and building multiple convolutional neural networks.

## Preliminary Findings

We have encouraging early results. Below are 10 randomly selected pairs of starting and ending frames (e.g. one frame after another). The starting frames on the left have the bounding box originally given as an input (green). The ending frames on the right have the ground truth bounding box (green) and the bounding box predicted by our model (red).

![Vid_1 Start](./readme_imgs/start_1.jpg)
![Vid_1 End](./readme_imgs/end_1.jpg)

![Vid 2 Start](./readme_imgs/start_21.jpg)
![Vid 2 End](./readme_imgs/end_21.jpg)

![Vid 3 Start](./readme_imgs/start_41.jpg)
![Vid 3 End](./readme_imgs/end_41.jpg)

![Vid 4 Start](./readme_imgs/start_61.jpg)
![Vid 4 End](./readme_imgs/end_61.jpg)

![Vid 5 Start](./readme_imgs/start_81.jpg)
![Vid 5 End](./readme_imgs/end_81.jpg)

![Vid 6 Start](./readme_imgs/start_108.jpg)
![Vid 6 End](./readme_imgs/end_108.jpg)

![Vid 7 Start](./readme_imgs/start_121.jpg)
![Vid 7 End](./readme_imgs/end_121.jpg)

![Vid 8 Start](./readme_imgs/start_141.jpg)
![Vid 8 End](./readme_imgs/end_141.jpg)

![Vid 9 Start](./readme_imgs/start_161.jpg)
![Vid 9 End](./readme_imgs/end_161.jpg)

![Vid 10 Start](./readme_imgs/start_181.jpg)
![Vid 10 End](./readme_imgs/end_181.jpg)


## Error Analysis

In plotting the actual versus predicted coordinates below for a random sample of 500 images, we can get a sense of how our network is learning. At the top-left we have x0, top right y0, bottom-left x1, and bottom-right x1. These correspond to the upper left corner (x0, y0) and bottom right corner (x1, y1) of the bounding. The kernal density estimates below show that we are on average predicting fairly well (as seems to also be indicated by the images above), but still have some variabilty in how well those predictions are lining up to the ground truth. 


![Error X Dimension](./readme_imgs/x0.png)
![Error Y Dimension](./readme_imgs/y0.png)

![Error X Dimension](./readme_imgs/x1.png)
![Error Y Dimension](./readme_imgs/y1.png)

Moving forward, we hope to continue improving the object tracker through alternative architectures and larger networks.
