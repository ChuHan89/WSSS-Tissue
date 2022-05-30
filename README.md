# Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels
![outline](workflow.png)

## Introduction
The code of:

**Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels, Chu Han, Jiatai Lin, Jinhai Mai, Yi Wang, Qingling Zhang, Bingchao Zhao, Xin Chen, Xipeng Pan, Zhenwei Shi, Zeyan Xu, Su Yao, Lixu Yan, Huan Lin, Xiaomei Huang, Changhong Liang, Guoqiang Han, Zaiyi Liu, Medical Image Analysis, 2022.**[[Paper]](https://doi.org/10.1016/j.media.2022.102487)

We present a tissue semantic segmentation model for histopathology images using only patch-level classification labels, which greatly saves the annotation time for pathologists. Multi-layer pseudo-supervision with progressive dropout attention is proposed to reduce the information gap between patch-level and pixellevel labels. And a classification gate mechanism is introduced to reduce the false-positive rate.


## Dataset
We have released both datasets via Google Drive ([LUAD-HistoSeg](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing) and [BCSS-WSSS](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing).).
We would like to thank Amgad et al. for this excellent dataset. The original BCSS dataset can be download at this link [BCSS-link](https://github.com/PathologyDataScience/CrowdsourcingDataset-Amgadetal2019).
## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@misc{han2021multilayer,
      title={Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels}, 
      author={Chu Han and Jiatai Lin and Jinhai Mai and Yi Wang and Qingling Zhang and Bingchao Zhao and Xin Chen and Xipeng Pan and Zhenwei Shi and Xiaowei Xu and Su Yao and Lixu Yan and Huan Lin and Zeyan Xu and Xiaomei Huang and Guoqiang Han and Changhong Liang and Zaiyi Liu},
      year={2021},
      eprint={2110.08048},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
```
@article{HAN2022102487,
title = {Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation using Patch-level Classification Labels},
journal = {Medical Image Analysis},
pages = {102487},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102487},
url = {https://www.sciencedirect.com/science/article/pii/S1361841522001347},
author = {Chu Han and Jiatai Lin and Jinhai Mai and Yi Wang and Qingling Zhang and Bingchao Zhao and Xin Chen and Xipeng Pan and Zhenwei Shi and Zeyan Xu and Su Yao and Lixu Yan and Huan Lin and Xiaomei Huang and Changhong Liang and Guoqiang Han and Zaiyi Liu}}
```
