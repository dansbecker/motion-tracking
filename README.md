# Motion-Tracking 

Inspired by [recent advances in offline object tracking](http://arxiv.org/pdf/1604.01802v1.pdf), this project aims to recreate, further understand, and improve upon these recent advancements. The end goal is to be able to more accurately track a given object of interest from one video frame to the next. 

Thus far, this project has involved recreating the results of these recent advances as well as exploring the results of the generated models. This has involved collecting data from the [ALOV300++ Dataset](http://www.alov300.org/) as well as [ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/), augmenting that data through random croppings, and building multiple convolutional neural networks. 
 
## Preliminary Findings 

At this point, we've gotten some encouraging results. Below are 10 randomly selected pairs of starting and ending frames (e.g. one frame after another). The starting frames on the left have the originally given bounding box (green), and the ending frames on the right have the ground truth bounding box (green) as well as the bounding box predicted by our net (red). 

<img  width="50"\>
<img src="./readme_imgs/start_1.jpg" width="175"\>
<img src="./readme_imgs/end_1.jpg" width="175"\>
<img  width="50"\>
<img src="./readme_imgs/start_21.jpg" width="175"\>
<img src="./readme_imgs/end_21.jpg" width="175"\>

<img  width="50"\>
<img src="./readme_imgs/start_41.jpg" width="175"\>
<img src="./readme_imgs/end_41.jpg" width="175"\>
<img  width="50"\>
<img src="./readme_imgs/start_61.jpg" width="175"\>
<img src="./readme_imgs/end_61.jpg" width="175"\>

<img  width="50"\>
<img src="./readme_imgs/start_81.jpg" width="175"\>
<img src="./readme_imgs/end_81.jpg" width="175"\>
<img  width="50"\>
<img src="./readme_imgs/start_108.jpg" width="175"\>
<img src="./readme_imgs/end_108.jpg" width="175"\>

<img  width="50"\>
<img src="./readme_imgs/start_121.jpg" width="175"\>
<img src="./readme_imgs/end_121.jpg" width="175"\>
<img  width="50"\>
<img src="./readme_imgs/start_141.jpg" width="175"\>
<img src="./readme_imgs/end_141.jpg" width="175"\>

<img  width="50"\>
<img src="./readme_imgs/start_161.jpg" width="175"\>
<img src="./readme_imgs/end_161.jpg" width="175"\>
<img  width="50"\>
<img src="./readme_imgs/start_181.jpg" width="175"\>
<img src="./readme_imgs/end_181.jpg" width="175"\>

