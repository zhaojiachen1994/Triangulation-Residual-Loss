## Triangulation Residual Loss


### Introduction
Code for paper: Triangulation Residual Loss for Data-efficient 3D Pose Estimation. TR loss enables self-supervision with global 3D geometric consistency by minimizing the smallest singular value of the triangulation matrix. Particularly, TR loss aims to minimize the weighted sum of distances from the current 3D estimate to all view rays, so that the view rays will converge to a stable 3D point.



### Usage
- Install mmcv==1.7.0 and mmpose==0.29.0 following the [guideline](https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md)
- clone this project and install the requirements.


### Datasets

- [Calm21(MARS) dataset](https://neuroethology.github.io/MARS/dataset.html): images could be download from [here](https://neuroethology.github.io/MARS/dataset.html), the annotations could be downloaded from [here](https://drive.google.com/drive/folders/1r4LvGSjYGzQyRl0UBUvmfuEw-o2nmfl0?usp=drive_link)

- Dannce and THM datasets: The used images and annotations could be download from [here](https://drive.google.com/drive/folders/1r4LvGSjYGzQyRl0UBUvmfuEw-o2nmfl0?usp=drive_link)

- [Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php) is formulated in COCO annotation form.

- Download the dataset to your loacal computer, then modify the 'data_root' in the config file to the downloaded path.

### Train and evaluate



The pretrained backbone and models could be downloaded from here.


### How to apply TR loss to your model?
The core code of TR loss is in TRL/models/heads/triangulate_head.py as following:

```python
u, s, vh = torch.svd(A.view(-1, 4)) # A is the matrix defined in (13)
res_triang = s[-1] # res_triang is the TR Loss
```
Then add the TR loss to your final losses and perform gradient backpropagation.

### Acknowledgements
 [1] https://github.com/zhezh/adafuse-3d-human-pose

 [2] https://github.com/karfly/learnable-triangulation-pytorch

 [3] https://github.com/luminxu/Pose-for-Everything