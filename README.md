# Greatness in Simplicity: Unified Self-cycle Consistency for Parser-free Virtual Try-on, NeurIPS 2023
**Official code for paper "[Greatness in Simplicity: Unified Self-cycle Consistency for Parser-free Virtual Try-on](https://arxiv.org/abs/)"**

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**The `test code` has been released, the `training code` will be released after the paper has been accepted.**

![image](https://github.com/anony-conf/USC-PFN/blob/main/figures/compare.jpg?raw=true)

The pursuit of an efficient lifestyle has been stimulating the development of image-based virtual try-on. However, generating high-quality virtual try-on images remains challenging due to the inherent difficulties such as modeling non-rigid garment deformation and unpaired garment-person images. Recent groundbreaking formulations, including in-painting, cycle consistency, and in-painting-based knowledge distillation, have enabled self-supervised generation of try-on images. Nevertheless, these methods require disentangling different garment domains in the try-on result distribution via an assistance of "teacher knowledge" or dual generators. Due to the possible existence of irresponsible prior knowledge in the pretext task, such multi-model cross-domain pipelines may act as a significant bottleneck of main generator (e.g., "student model," CNN_2 of DCTON) in downstream task, leading to reduced try-on quality. Additionally, current garment deformation methods are unable to mimic the natural interaction between the garment and the human body in the real world, resulting in unrealistic alignment effects. To tackle these limitations, we present a new Unified Self-Cycle Consistency for Parser-Free virtual try-on Network (USC-PFN), which enables the robust translation between different garment domains using only a single generator and realistically mimics the non-rigid geometric deformation of garments in the real world. Specifically, we first propose a self-cycle consistency architecture with a round mode that uses only unpaired garment-person images as inputs for virtual try-on, which effectively shakes off irresponsible prior knowledge. Markov Random Field is first formulated for more natural and realistic garment deformation. Moreover, USC-PFN can employ general generator for self-supervised cycle training. Experiments demonstrate that our method achieves SOTA performance on a popular virtual try-on benchmark. 

[[Paper]](https://arxiv.org/abs/)       [[Supplementary Material]](https://github.com/)

[[Checkpoints for Test]](https://drive.google.com)

[[Training_Data]](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing)
[[Test_Data]](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing)

[[VGG_Model]](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing)

## Our Environment
- anaconda3

- pytorch 1.6.0

- torchvision 0.7.0

- cuda 11.7

- cupy 8.3.0

- opencv-python 4.5.1
 
- python 3.6

1 tesla V100 GPU for training; 1 tesla V100 GPU for test


## Installation
`conda create -n sccn python=3.6`

`source activate sccn     or     conda activate sccn`

`conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=11.7 -c pytorch`

`conda install cupy     or     pip install cupy==8.3.0`

`pip install opencv-python`

`git clone https://github.com/anony-conf/USC-PFN.git`

## Inference

## Run the demo
1. cd USC-PFN
2. First, you need to download the checkpoints from [checkpoints](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing) and put the folder "USC-PFN" under the folder "checkpoints". The folder "checkpoints/USC-PFN" shold contain "warp_model_final.pth" and "gen_model_final.pth". 
3. The "dataset" folder contains the demo images for test, where the "test_img" folder contains the person images, the "test_clothes" folder contains the clothes images, and the "test_edge" folder contains edges extracted from the clothes images with the built-in function in python (We saved the extracted edges from the clothes images for convenience). 'demo.txt' records the test pairs. 
4. During test, a person image, a clothes image and its extracted edge are fed into the network to generate the try-on image. **No human parsing results or human pose estimation results are needed for test.**
5. To test with the saved model, run **test.sh** and the results will be saved in the folder "results".
6. **To reproduce our results from the saved model, your test environment should be the same as our test environment, especifically for the version of cupy.** 

![image](https://github.com/anony-conf/USC-PFN/blob/main/figures/results.jpg?raw=true)

## Dataset
1. [VITON](https://github.com/xthan/VITON) contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our saved model is trained on the VITON training set and tested on the VITON test set.
2. To train from scratch on VITON training set, you can download [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing).
3. To test our saved model on the complete VITON test set, you can download [VITON_test](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing).

## Inference

## Evaluation SSIM (Structural Similarity) and FID (Fréchet Inception Distance)

The results for computing SSIM is **same-clothes reconstructed results** (paired setting), FID is **different-clothes reconstructed results** (unpaired setting). 

### SSIM score
  1. Use the pytorch SSIM repo. https://github.com/Po-Hsun-Su/pytorch-ssim
  2. Normalize the image (img/255.0) and reshape correctly. If not normalized correctly, the results differ a lot. 
  3. Compute the score with window size = 11, the SSIM score should be 0.91.

### FID score
  1. Use the pytorch inception score repo. https://github.com/toshas/torch-fidelity
  2. Install FID use `pip install torch-fidelity`. Please strictly follow the procedure given in this repo.
  3. Compute the score, the FID score should be 10.47.
  
  ```CUDA_VISIBLE_DEVICES=0 python -m pytorch_fid path_results_A/ path_results_B/```


## Acknowledgement
Our code is based on the official implementation of [[PF-AFN](https://github.com/geyuying/PF-AFN)] and the unofficial implementation of [[SieveNet](https://github.com/levindabhi/SieveNet)]. If you use our code, please also cite their work as below.


## Citation
If our code is helpful to your work, please cite:
```
@article{du2023greatness,
  title={Greatness in Simplicity: Unified Self-cycle Consistency for Parser-free Virtual Try-on},
  author={Du, Chenghu and Wang, Junyin and Liu, Shuqing and Xiong, Shengwu},
  journal={Advances in Neural Information Processing Systems},
  publisher={Curran Associates, Inc.}
  pages={1--12},
  year={2023}
}
```

```
@inproceedings{ge2021parser,
  title={Parser-free virtual try-on via distilling appearance flows},
  author={Ge, Yuying and Song, Yibing and Zhang, Ruimao and Ge, Chongjian and Liu, Wei and Luo, Ping},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={8485--8493},
  year={2021}
}

@inproceedings{jandial2020sievenet,
  title={Sievenet: A unified framework for robust image-based virtual try-on},
  author={Jandial, Surgan and Chopra, Ayush and Ayush, Kumar and Hemani, Mayur and Krishnamurthy, Balaji and Halwai, Abhijeet},
  booktitle={Proceedings of the IEEE/CVF winter conference on applications of computer vision},
  pages={2182--2190},
  year={2020}
}
```
## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.
