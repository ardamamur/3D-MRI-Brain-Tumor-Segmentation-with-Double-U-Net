# 3D MRI Brain Tumor Segmentation with Double U-Net

## Abstract
Artificial Intelligence and deep networks had a great impact on medical imaging tech-
nologies. The Brain Tumor Segmentation (BraTS) [1, 5, 2] Challenge, for example, ac-
celerated progress in this task significantly. The standard go-to model for segmentation
in medical imaging is the U-Net [7]. Whilst being already a good network, modifications
for further improvement still exist. One of them is the Double U-Net [4] that promises
to be more accurate on smaller segmentation objects in the 2-dimensional case.
## Requirements
### Overview
The quantitative evaluation of brain tumors is crucial in the oncology, since the outcome
can be used both for diagnosis and treatment. Automatic segmentation is attractive and
effective in this field because it enables faster, ideally more objective (assuming proper
bias controlling), and more precise definition of pertinent tumor characteristics, such as
the volume of its sub-regions (which is important for treatment procedures). In this
project, we will extend the double U-Net architecture for 3D data. Or in other words,
we will be combining the 3D U-Net [3] architecture with the 2D Double U-Net [4].
In addition, we will also run one of the state-of-the-art models which use autoencoder
regularization [6] and compare them. The training and evaluation will be done on the
BraTS dataset [1, 5, 2].
### Methodology
Our aim is to combine both the 3D U-Net [3] and the Double U-Net [4] using PyTorch
and implement possible and sensible adaptations when necessary (due to domain and
data change) on the BraTS dataset [1, 5, 2]. Also, we will compare them with one of
the existing state-of-the-art which is using autoencoder regularization [6].
### Dataset
For this project, we need a dataset of 3D MRI Brain images. For that we decided to
use the Brain Tumor Segmentation (BraTS) Challenge dataset. BraTS has always been
focusing on the evaluation of state-of-the-art methods for the segmentation of brain
tumors in multi-modal magnetic resonance imaging (MRI) scans.
## Reference Repositories
[BraTS20 3DUnet 3D Auto Encoder](https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder/notebook) 

[Medical Net](https://github.com/Tencent/MedicalNet) 

https://github.com/doublechenching/brats_segmentation-pytorch

[MedicalZooPytorch](https://github.com/black0017/MedicalZooPytorch)


## License

[MIT](https://choosealicense.com/licenses/mit/)
