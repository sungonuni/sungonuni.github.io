---
permalink: /
title: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I'm a third-year Ph.D student of [Efficient Computing Lab](https://sites.google.com/view/eh-p) in the Department of Computer Science & Engineering at [POSTECH](https://www.postech.ac.kr), advised by Prof. Eunhyeok Park. Before joining POSTECH, I completed my B.S. in Department of Computer Science & Engineering in [Kyung Hee University](https://www.khu.ac.kr).

I'm currently focusing on Efficient AI, particularly in enhancing **Memory Efficiency** and **Computation Acceleration** during model **training** and **inference** of various models (Vision, LLM, Video Generation, etc.) via **Quantization** and **Low-rank Approximation**.

Research Keywords
-----
- Fast and Memory Efficient Training ([A01](#A01), [A02](#A02), [C03](#C03))
- Fast and Memory Efficient Inference ([C01](#C01), [C02](#C02), [P02](#P02), [P04](#P04))
- Parameter Efficient Fine-tuning of LLMs ([P03](#P03))
- HW - SW codesign via kernel optimization ([P02](#P03), [P04](#P04))



News
-----
- [Nov. 03, 2025] I've joined [**AMD**](https://www.amd.com/) as a Research Associate. 
- [Oct. 21, 2025] I've been selected as a **Winner** of [**Qualcomm Innovation Fellowship Korea**](https://www.qualcomm.com/research/university-relations/innovation-fellowship).
- [Oct. 01, 2025] I've been selected for the [**BK21 Outstanding Graduate Student International Training Scholarship**](https://bk21four.nrf.re.kr/) by Ministry of Science and ICT, South Korea.
- [Mar. 02, 2025] 1 paper has been accepted to [**ICML 2025**](https://icml.cc//).
- [Feb. 25, 2025] 1 paper has been accepted to [**CVPR 2025**](https://cvpr.thecvf.com/).
- [Oct. 28, 2024] 1 paper has been accepted to [**WACV 2025**](https://wacv2025.thecvf.com/).


Publications
-----
<a id="A02"></a>
- **[A02]** [AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation](https://arxiv.org/abs/2604.02525)  
**Seonggon Kim**, Alireza Khodamoradi, Pranathi Vasireddy, Kristof Denolf, Eunhyeok Park  
arXiv 2604.

<a id="C03"></a>
- **[C03]** [HOT: Hadamard-based Optimized Training](https://arxiv.org/abs/2503.21261)  
**Seonggon Kim**, Juncheol Shin, Seung-taek Woo, Eunhyeok Park  
Computer Vision and Pattern Recognition (**CVPR 2025**), Nashville.

<a id="C02"></a>
- **[C02]** [Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](https://arxiv.org/abs/2505.23651)  
Juncheol Shin, Minsang Seok, **Seonggon Kim**, Eunhyeok Park  
International Conference on Machine Learning (**ICML 2025**), Vancouver.

<a id="C01"></a>
- **[C01]** [PTQ4VM: Post-training Quantization for Visual Mamba](https://arxiv.org/abs/2412.20386)  
Younghyun Cho\*, Changhun Lee\*, **Seonggon Kim**, Eunhyeok Park  
Winter Conference on Applications of Computer Vision (**WACV 2025 <span style="color:red">Oral</span>**), Tucson.

<a id="A01"></a>
- **[A01]** [HLQ: Fast and Efficient Backpropagation via Hadamard Low-rank Quantization](https://arxiv.org/abs/2406.15102)  
**Seonggon Kim**, Eunhyeok Park  
arXiv 2406.


Project
-----

<a id="P05"></a>
- **[P05]** Low-Precision Training on AMD GPU architechture,	Jun. 2024 - Jun. 2025   
Advanced Micro Devices
  - Conducted research on Low-Precision (MXFP4) Training on AMD CDNA4 architecture. 
  - Implemented **ROCm kernel** for training acceleration. 
  - Achieved **3.6X memory compression, 1.46X acceleration** over BF16. 
  - Published the result on [A02](#A02)


<a id="P04"></a>
- **[P04]** Solutions for slow SVD approximation problem of KV Cache compression,	Jun. 2024 - Jun. 2025   
POSTECH
  - Conducted research on slow SVD approximation problem during KV Cache compression.
  - Implemented **CUDA kernel** for SVD approximation acceleration.    
  - Achieved **5.1X acceleration and 95% similarty** with exact SVD operation on GSM8K using LLaMA3-8B. 
  - The result of research is under review.

<a id="P03"></a>
- **[P03]** Solutions for misaligned weight initialization problem of LoRA finetuning,	Jun. 2024 - Jun. 2025   
POSTECH
  - Conducted research on Effective LoRA weight initialization.    
  - Achieved **99% accuracy of full fine-tuning** with only rank 16 on GSM8K using LLaMA3-8B. 
  - The result of research is under review.

<a id="P02"></a>
- **[P02]** GEMV Accelerator for LLM inference on Intel Gaudi-2,	Jun. 2024 - Jun. 2025   
Naver & Intel Joint Research Center  
  - Conducted research on LLM’s fast inference on Intel Gaudi-2 architecture.    
  - Implemented custom **GEMV kernel for Gaudi** with TPC-C language.  
  - Transplanted LUT Quantization from CUDA to Gaudi TPC.  

<a id="P01"></a>
- **[P01]** Solutions for Efficient training on limited GPU environment,	Jun. 2023 – Current  
Ministry of Science and ICT of Korea
  - Conducted research on Low-Precision (INT4) Training on limited GPU environment.  
  - Implemented **CUDA kernel** for training acceleration.
  - Achieved up to **75% memory savings and a 2.6X acceleration** over FP32.
  - Published the result on [A01](#A01), [C03](#C03)


Experience
-----
- **Research Associate**, Nov. 2025 - May. 2026  
Advanced Micro Devices, Longmont, CO, USA

- **Software Engineer Intern**, Jul. 2022 - Feb. 2023   
Spirent Communications, San Jose, CA, USA

- **Software Engineer Intern**, Feb. 2022 - Jun. 2022  
Common Computer, Seoul, Korea

- **Research Intern**, Mar. 2021 - Dec. 2021  
SI Analytics, Daejeon, Korea


Awards & Honors
-----
- Qualcomm Innovation Fellowship Korea, **Final Winner**, Oct. 2025

- BK21 Outstanding Graduate Student International Training Scholarship, **Selected Graduate Student**, Oct. 2025

- CVPR 2021 Earthvision workshop, Land Cover Classification Challenge, **5th Prize**  
CVPR, Jun. 2021


Education
-----
- **M.S/Ph.D.** in Computer Science and Engineering, [POSTECH](https://www.postech.ac.kr)  
Sep. 2023 - Present

- **B.S.** in Computer Science and Engineering, [Kyung Hee University](https://www.khu.ac.kr)  
Feb. 2017 - Aug. 2023

Teaching Experience
-----
- Teaching Assistant, Sep. 2026 - Dec. 2026  
CSED342: Artificial Intelligence, [POSTECH](https://www.postech.ac.kr) 

- Teaching Assistant, Mar. 2025 - June. 2025  
CSED311: Computer Architecture, [POSTECH](https://www.postech.ac.kr) 
