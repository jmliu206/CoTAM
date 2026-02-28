# WHEN MLLMS MEET COMPRESSION DISTORTION: A CODING PARADIGM TAILORED TO MLLMS (CoTAM)

> 
> **Authors:** Jinming Liu, Zhaoyang Jia, Jiahao Li, Bin Li, Xin Jin, Wenjun Zeng, Yan Lu 
> 
> 
> 
> **Institutions:** Shanghai Jiao Tong University, Microsoft Research Asia, Eastern Institute of Technology 
> 
> 

This is the official PyTorch implementation of the ICLR 2026 paper **"WHEN MLLMS MEET COMPRESSION DISTORTION: A CODING PARADIGM TAILORED TO MLLMS"**.

---

## 📢 News

* **[Feb 2026]** Our paper has been accepted to **ICLR 2026**! 


* **[Coming Soon]** The full training and evaluation code, along with pre-trained models, will be released in **April 2026**. 

---

## 💡 Abstract & Motivation

The increasing deployment of powerful Multimodal Large Language Models (MLLMs) on cloud platforms urgently requires effective compression techniques to efficiently transmit signal inputs from edge devices with minimal bandwidth.

However, conventional image codecs are engineered for the Human Visual System (HVS) and are ill-suited for the diverse downstream tasks of MLLMs. Through systematic analysis, we discovered a crucial insight: **Compression distortion unevenly impacts different-level image features, leading to varying effects on MLLMs' downstream tasks depending on their feature-level reliance**. Specifically, cross-level features (e.g., needed for counting objects) are highly susceptible to compression artifacts, as the disruption of low-level information breaks coherent high-level semantics.


---

## ⏳ Getting Started (Coming April 2026)

Instructions for setting up the environment, running the encoding/decoding pipeline, and reproducing the benchmarks will be provided here upon code release.

### Prerequisites

* Python 3.x
* PyTorch




---

## 🔗 Citation

If you find our work or this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{liu2026when,
  title={WHEN MLLMS MEET COMPRESSION DISTORTION: A CODING PARADIGM TAILORED TO MLLMS},
  author={Liu, Jinming and Jia, Zhaoyang and Li, Jiahao and Li, Bin and Jin, Xin and Zeng, Wenjun and Lu, Yan},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026}
}

```

## 📧 Contact

For any questions regarding the paper or the upcoming code release, please contact Jinming Liu at [jmliu206@sjtu.edu.cn](mailto:jmliu206@sjtu.edu.cn) or Xin Jin at [jinxin@eitech.edu.cn](mailto:jinxin@eitech.edu.cn).

