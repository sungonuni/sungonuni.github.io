---
layout: archive
title: "Training with Half Memory: Vision Model Training via Hadamard Low-Rank Adaptation"
permalink: /EVMT-HLA/
author_profile: true
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


## **One-Sentence Teaser**

By using Hadamard low-rank adaptation, we reduce memory usage by **up to 50%**—while **preserving pretraining loss and accuracy in DeiT**, and **even improving performance in SimCLR, BYOL.**


## **TL;DR**
<div align="center">
<img src="/images/EVMT-HLA/concept.png" style="width:600px;">
</div>



- **Problem:** Various vision models—including self-supervised learning (SSL) frameworks such as SimCLR [1] and BYOL [2], as well as supervised models like DeiT [3]—often suffer from memory constraints due to their large batch size requirements.
- **Idea:** Recent studies like Grokfast [4] and HOT [5] show that emphasizing low-frequency components of gradients during training can reduce overfitting to noise while enhancing a model’s ability to capture core data patterns. Inspired by this, we adopt Hadamard Low-rank Approximation (HLA) to gradient to get low frequency filtered gradient.
- **Result:** This approach cuts memory usage by half. On DeiT [3], performance is preserved; for SimCLR [1] and BYOL [2], it even leads to performance gains of +0.5 percentage points and +0.02 percentage points, respectively.

---

## **Problem: Memory limitation on Vision model training**

Over the past decade, deep learning has transformed computer vision—but it came at a cost: **massive datasets** and **even more massive GPUs**. Popular contrastive learning-based vision models like SimCLR [1] and BYOL [2] have shown that self-supervised learning can match or even surpass supervised learning, especially when labeled data is scarce. Similarly, data-efficient architectures like DeiT [3] aim to shrink the data budget without hurting performance.

But here's the catch: these methods are not computationally efficient. Contrastive learning models demand large batch sizes—sometimes in the thousands—to work well. And for DeiT [3], despite being “data-efficient,” still consumes a surprising amount of memory during training.

This creates a bottleneck: hit a **GPU memory wall**. Researchers and practitioners with limited hardware can't easily train or experiment with these models. This motivates the development of approaches that alleviates memory bottlenecks during training, while retaining the benefits of data augmentation. 

## Generalize Better, Train Lighter: Gradient Filtering with HLA

Recent research suggests that not all parts of the gradient are equally useful. Studies like Grokfast [4]  have shown that filtering out high-frequency components of the gradient and emphasizing the low-frequency parts can help models generalize better. These “low-pass” gradients highlight coarse structural patterns in the data, reducing overfitting and leading to smoother, more stable training dynamics.

That’s where our work comes in. We extend this idea to real-world computer vision tasks using a technique called **Hadamard Low-rank Approximation (HLA)**. By projecting gradients into a low-rank Hadamard space, we retain the essential low-frequency signals while discarding noisy, high-frequency clutter. The result? The GPU memory consumption decreases 50% less while preserving the model performance, and **even increase the performance in certain models.**

## What is Hadamard Low-rank Approximation?

### Hadamard Transformation

A **Hadamard matrix** is a square orthogonal matrix composed entirely of +1 and –1 entries. The **Hadamard Transform (HT)** maps an input vector linearly using the Hadamard matrix. It can be interpreted as a generalized version of the discrete Fourier transform over real numbers.

The base Hadamard matrix is defined as:

$$
H_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad H_n = H_1 \otimes H_{n-1}
$$

Given an input vector $`X \in {R}^{2^d}`$, the Hadamard Transform can be computed via matrix multiplication:

$$
Y = X \cdot H_d
$$

This naive Hadamard method has a computational complexity of ${O}(n^2)$.

However, using only in-place addition and subtraction operations (without explicit matrix multiplication), the **Fast Walsh–Hadamard Transform (FWHT)** [6] computes the transform much more efficiently, with $O(nlogn)$.

### Hadamrad Low-rank Approximation

Since the Hadamard Transform (HT) is an orthogonal transform similar to the discrete Fourier transform, it allows us to filter out high-frequency noise by selecting only a subset of the transformed vector. This selective filtering is referred to as **Hadamard Low-rank Approximation (HLA)**. Depending on how the matrix operations are amortized during implementation, HLA can be categorized into **Internal** and **External** types.

**Internal HLA** amortizes the approximation within the matrix operation itself. Let the tensor $P \in \mathbb{R}^{M \times K}$  and another tensor $R \in \mathbb{R}^{K \times N}$  are given, the HLA-transformed weight update takes the form:

$$
\hat{R} = (P \cdot \hat{H}^T) \cdot (\hat{H} \cdot S)
$$

where $\hat{H} \in \mathbb{R}^{r \times K}$ is the truncated (rank-$r$) Hadamard projection matrix,

**External HLA**, in contrast, requires the inverse transform to be applied explicitly after the main operation:

$$
\hat{R} = \hat{H}^T \cdot (H \cdot P \cdot S)
$$

## **Our Approach: Efficient Vision Model Training-HLA**

In this work, we propose **EVMT-HLA**, a simple yet powerful technique that improves both the generalization and memory efficiency of vision models—by moving the backpropagation process into the **frequency domain**.

This approach allows us to **emphasize informative, low-frequency components while suppressing noisy, high-frequency ones—leading to better learning dynamics and significantly lower memory usage during training.**
<div align="center">
<img src="/images/EVMT-HLA/architecture.png" style="width:600px;">
</div>

### Low-Rank Matrix Multiplication via HLA

To reduce computation and memory usage during backpropagation, we selectively apply **Hadamard Low-rank Approximation (HLA)** to the weight gradient calculation.

Specifically:

- **Forward pass**
    - The **activation gradient** is transformed using HT, allowing the subsequent matrix multiplication with the weight and output gradient to occur entirely in the frequency domain. This transformation is efficient thanks to the **Fast Walsh–Hadamard Transform (FWHT)** [6] algorithm, which introduces only minimal overhead while preserving full-rank gradient fidelity.
- **Backward pass**
    - We apply HLA again to the output gradient tensor. This projection compresses the tensor before matrix multiplication, **directly reducing the computational cost** of the gradient update. The HLA operation is defined as follows:
        
        $$
        HLA(x)=H_r \times x 
        $$
        
        where $H_R=H[: r, :]$  is a partial matrix selecting the first r rows from the Hadamard matrix H. r is amount of low frequency components to preserve.
        
    - Two types of gradient calculations are performed: activation gradient and weight gradient. Using a linear layer as an example, we can explain as follows. Assume there is a weight $w ∈ R^{O×I},$ output gradient $g_y ∈ R^{L×O}$ , activation tensor $x ∈ R^{L×I}$ . Here, $L$ represents the spatial length, $O$  the output channel, and $I$ the input channel. The weight gradient $g_w ∈ R^{O×I}$ is calculated as the outer product of the output gradient and activation tensor:
        
        $$
        g_w = {g_y}^T \times x 
        $$
        
        In our method, HLA is selectively applied to the $g_w$ path:
        
    
    $$
    g_w = HLA(g_y) \times HLA(x)
    $$
    
    - **Note:** Activation tensors are pre-processed into low-rank representations using HLA. These approximated tensors are cached, which reduces the memory footprint during the backward pass.

## Experiment Setting

We selected SimCLR [1], BYOL [2], and DeiT [3] as target models for our experiments, and set the batch size to 256—the largest size that could be accommodated within the memory constraints of an RTX 3090.

## **Experiments & Key Results**

- **Standard Floating-point (FP) Training V.S. EVMT-HLA**
    - For SimCLR [1], our EVMT-HLA showed an **improvement of approximately 0.5%p**, and for other models, the performance remained **nearly identical to FP training.**
    <div align="center">
    <img src="/images/EVMT-HLA/main_results.png" style="width:550px;">
    </div>
        
- **Memory reduction**
    - By applying EVMT-HLA with a 50% rank reduction ratio, we observe a **memory reduction of more than 50%** compared to FP training for all models.
    <div align="center">
    <img src="/images/EVMT-HLA/memory_reduction.png" style="width:400px;">
    </div>

    
- **Convergence Speed**
    - When using EVMT-HLA, we observe **faster convergence** compared to FP training. The figure below shows the training loss curve of SimCLR [1].
    <div align="center">
    <img src="/images/EVMT-HLA/convergence.png" style="width:600px;">
    </div>

- **Dropout V.S. EVMT-HLA**
    - Despite the reduction of tensor size by half during gradient computation, the performance was preserved—and even improved in the case of SimCLR [1]. We hypothesize that this is due to the presence of meaningful gradient components in the low-frequency spectrum, as well as a potential **regularization effect.** To validate this hypothesis, we employed dropout as a form of regularization, which randomly selects tensor elements to be dropped. We compared models trained with a 50% dropout ratio to those trained using EVMT-HLA with 50% preservation of low-frequency components.
    - As a result, while applying dropout led to nearly a 50% drop in performance compared to FP training, EVMT-HLA maintained performance comparable to FP, and in some cases even outperformed it. These findings suggest that **low frequency component of gradient successfully captures the critical components for learning, and the observed improvement can be attributed to its effective regularization effect.**
    <div align="center">
    <img src="/images/EVMT-HLA/dropout.png" style="width:550px;">
    </div>

    
- **Rank Reduction Ratio**
    - In EVMT-HLA, the rank reduction ratio determines the percentage of the tensor size that is reduced before weight gradient calculation. As this ratio increases, more information is discarded, resulting in a decline in performance. As an example, we conducted a sweep of reduction ratios from 90% to 50% in SimCLR [1] and measured the performance.
    - We found that up to a 70% reduction ratio, the preserved low-frequency components still retain essential information. However, when the ratio drops below 70%, significant information loss occurs.
    <div align="center">
    <img src="/images/EVMT-HLA/memory_reduction.png" style="width:400px;">
    </div>

    

## **Takeaways**

- **Takeaway 1:**
    
    The low-frequency components of gradients play a key role in **improving generalization across a wide range of vision models.**
    
- **Takeaway 2:**
    
    **Hadamard Low-rank Approximation (HLA)** provides an efficient way to selectively preserve only those low-frequency signals during backpropagation—**reducing memory usage by up to ~50%** while maintaining, or even improving, performance in vision tasks.
    

## Limitations

- **Effect of Batch Size**
    
    Batch size is a well-known factor that influences both generalization and convergence stability. When used alongside EVMT-HLA, especially with small batch sizes, the combined regularization effects may lead to excessive information loss or underfitting. Further investigation is needed to understand and balance these effects.
    
- **Extending to More Models and Datasets**
    
    We plan to explore how EVMT-HLA performs on a wider variety of vision architectures (e.g., ViT, Swin, ConvNeXt) and datasets beyond CIFAR-10. This will help validate the robustness and scalability of the approach in more complex, real-world scenarios.
    

## Quick Start

https://github.com/sungonuni/EVMT-HLA

- **DeiT**
    - First, clone the repository locally:
        
        ```
        git clone <https://github.com/facebookresearch/deit.git>
        ```
        
    - Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):
        
        ```
        conda install -c pytorch pytorch torchvision
        pip install timm==0.3.2
        ```
        
    - Run training code.
        
        ```
        ./deit_train_repo.sh
        ```
        
- **SimCLR**
    - Clone the repository of SimCLR locally:
        
        ```python
        git clone <https://github.com/sthalles/SimCLR.git>
        ```
        
    - Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):
        
        ```
        conda install -c pytorch pytorch torchvision
        pip install timm==0.3.2
        ```
        
    - Run training code.
        
        ```
        ./simclr_train.sh
        ```
        

## References

[1] A simple framework for contrastive learning of visual representations.

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. 2020. *Advances in Neural Information Processing Systems*, 33.

[2] Bootstrap your own latent: A new approach to self-supervised learning.

Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z. D., Azar, M. G., Piot, B., Kavukcuoglu, K., Munos, R., & Valko, M. 2020.  *Advances in Neural Information Processing Systems*, 33.

[3] Training data-efficient image transformers & distillation through attention.

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & Jégou, H. 2021.  *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:10347–10357.

[4] Grokfast: Accelerated grokking by amplifying slow gradients.

Lee, J., Kang, B. G., Kim, K., & Lee, K. M. 2024.  *arXiv preprint arXiv:2405.20233*.

[5] HOT: Hadamard-based Optimized Training. 

Kim, S., Shin, J., Woo, S. T., & Park, E. (2025). arXiv preprint arXiv:2503.21261.

[6] Computation of the fast walsh-fourier transform.

John L Shanks.  IEEE Transactions on Computers, 100(5):457–459, 1969.