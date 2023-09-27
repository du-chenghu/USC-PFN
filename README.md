![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/du-chenghu/USC-PFN?style=social)](https://github.com/du-chenghu/USC-PFN)   &nbsp;
<img style="width:100px; float:left;" src="http://www.whut.edu.cn/images/whutlogo.png?raw=true">


<div align="center">

<h1>Greatness in Simplicity: Unified Self-cycle Consistency for Parser-free Virtual Try-on (NeurIPS 2023)</h1>

<div>
    <a href="https://github.com/du-chenghu/USC-PFN" target="_blank">Chenghu Du</a><sup>1</sup>,
    <a href="https://github.com/du-chenghu/USC-PFN" target="_blank">Junyin Wang</a><sup>1</sup>,
    <a href="https://github.com/du-chenghu/USC-PFN" target="_blank">Shuqing Liu</a><sup>4</sup>,
    <a href="https://github.com/du-chenghu/USC-PFN" target="_blank">Shengwu Xiong</a><sup>1,2,3,*</sup>
</div>

<div>
    <sup>1</sup>Wuhan University of Technology&emsp; <sup>2</sup>Sanya Science and Education Innovation Park
</div>
<div>
    <sup>3</sup>Shanghai AI Laboratory&emsp; <sup>4</sup>Wuhan Textile University
</div>

[Paper](https://arxiv.org/pdf/00.pdf) | [Supplementary Material](https://arxiv.org/pdf/00.pdf)
</br>

<strong>USC-PFN aims to transfer an in-shop garment onto a specific person.</strong>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="./imgs/head.gif?raw=true">
</div>

</div>

---

## Citation
If our code is helpful to your work, please cite:
```bibtex
@inproceedings{du2023greatness,
  title={Greatness in Simplicity: Unified Self-Cycle Consistency for Parser-Free Virtual Try-On},
  author={Du, Chenghu and Wang, Junyin and Liu, Shuqing and Xiong, Shengwu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
  pages={1--12},
}
```
---
## Todo
**The `test code` has been released, the `training code` will be released soon.**

- [ ] [2023-00-00] Release the **training script** for VITON dataset.
- [ ] [2023-00-00] Release the **pretrained model** for VITON dataset.
- [x] [2023-09-27] Release the **testing result** for VITON dataset.
- [x] [2023-09-23] Release the **testing script** for VITON dataset.
---

## Quick Access Results
The test pair and test results on VITON dataset are shown <a href="https://drive.google.com/file/d/1qxDdvBKIQisxxu2pYky5dkKEufj_bV_v/view?usp=sharing">this</a> and <a href="https://drive.google.com/file/d/1yYvBoTUqwQjllS9XELoDQFuqZXdgqzL3/view?usp=sharing">here</a>, from left to right are reference person, target clothes, try-on results of five baseline methods including **CP-VITON+ (CVPRW 2020), ACGPN (CVPR 2020), DCTON (CVPR 2021), RT-VITON (CVPR 2022), and USC-PFN.**

## Our Environment
- anaconda3

- pytorch 1.6.0

- torchvision 0.7.0

- cuda 11.7

- cupy 8.3.0

- opencv-python 4.5.1
 
- python 3.6


## Installation

```
pip3 install -r requirements.txt
```

**or**

`conda create -n uscpfn python=3.6`

`source activate uscpfn     or     conda activate uscpfn`

`conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=11.7 -c pytorch`

`conda install cupy     or     pip install cupy==8.3.0`

`pip install opencv-python`

`git clone https://github.com/du-chenghu/USC-PFN.git`

`cd ./USC-PFN/`

## Dataset
- [VITON](https://github.com/xthan/VITON) contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our saved model is trained on the VITON training set and tested on the VITON test set.

- To train from scratch on VITON training set, you can download [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing).

- To test our saved model on the complete VITON test set, you can download [VITON_test](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing).

## Inference
1. cd USC-PFN
```
cd USC-PFN-main
```
2. First, you need to download the [[Checkpoints for Test]](https://drive.google.com) and put these under the folder `checkpoints/`. The folder `checkpoints/` shold contain `ngd_model_final.pth` and `sig_model_final.pth`. 
3. Please put the test set of the dataset in the `dataset/`, i.e., the `dataset/` folder should contain the `test_pairs.txt` and `test/`.
4. To generate virtual try-on images, just run:
```
python test.py
```
5. The results will be saved in the folder `results/`.
> During inference, only a person image and a clothes image are fed into the network to generate the try-on image. **No human parsing results or human pose estimation results are needed for inference.**

> **To reproduce our results from the saved model, your test environment should be the same as our test environment.**

- Note that if you want to test **paired data (paired clothing-person images)**, please download the replaceable list <a href="https://drive.google.com/file/d/1yYvBoTUqwQjllS9XELoDQFuqZXdgqzL3/view?usp=sharing">here (test_pairs.txt)</a>.

## Training
```
Available soon...
```

## Evaluation - SSIM (Structural Similarity) and FID (Fréchet Inception Distance)

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

## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.
