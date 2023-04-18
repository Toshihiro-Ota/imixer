# iMixer: hierarchical Hopfield network implies an invertible, implicit and iterative MLP-Mixer

This is the official PyTorch implementation of iMixer created by

- [Toshihiro Ota](https://github.com/Toshihiro-Ota)
  - [![AnyTech](https://img.shields.io/badge/CyberAgent-Inc.-2c8c3c?style=plastic&labelColor=84bc2c)](https://www.cyberagent.co.jp/en/)
- [Masato Taki](https://scholar.google.com/citations?hl=en&user=3nMhvfgAAAAJ)
  - [![Rikkyo](https://img.shields.io/badge/Rikkyo-University-FFFFFF?style=plastic&labelColor=582780)](https://english.rikkyo.ac.jp)

The paper is available at [arXiv:2304.xxxxx](https://arxiv.org/abs/2304.xxxxx).

## Abstract

In the past years, Transformers have achieved great success not only in natural language processing but also in the field of computer vision. MLP-Mixer, which has spatial MLP token-mixing modules corresponding to the attention modules in Transformers, shows the competitive performance despite of the much less inductive bias inside the architecture. Recent studies on modern Hopfield networks suggest the correspondence between certain energy-based associative memory models and Transformers or MLP-Mixer, and shed some light on the theoretical background of the performance of Transformer-type architectures, whereas they are mostly based on either empirical or theoretical aspects. Inspired by the newly introduced hierarchical Hopfield network, in this paper we propose *iMixer*, a novel MLP-Mixer model which has an invertible, implicit, and iterative mixing module. In addition to the theoretical derivation of the model, we provide a practical algorithm for the inverted mixing module. We evaluate the model performance with various datasets on image classification tasks, and find that iMixer reasonably achieves the improvement compared to the baseline vanilla MLP-Mixer. The results show the effectiveness of our Hopfield-based approach and imply that the correspondence between the Hopfield networks and the Mixer models really works. As a byproduct, we also clarify that the proposed module can be regarded as an instance of the so-called implicit layer or the deep equilibrium model.

## Network Architecture

The overall architecture of iMixer as a form of MetaFormers:

<p align="center">
  @import "./img/metaformers.jpg" {width='60%' title='metaformers'}

iMLP module involved in iMixer as its token mixer:

<p align="center">
  @import "./img/imlp.jpg" {width='30%' title='imlp'}

## Model Configuration

| name | arch | Params | acc@1 (%) |
| --- | --- | --- | --- | --- | --- |
| iMixer-S | ```imixer_s16_224``` | 19M | 82.3 |
| iMixer-M | ```imixer_b16_224``` | xxM | 82.8 |
| iMixer-L | ```imixer_l16_224``` | xxM | 83.4 |

## Usage

### Requirements

- torch>=1.12.1
- torchvision
- timm==0.6.11
- matplotlib
- pillow
- scipy
- etc., see [requirements.txt](requirements.txt)

### Traning

Command line for training Sequencer models on ImageNet from scratch.

```
./distributed_train.sh 4 /path/to/cifar10 --dataset torch/cifar10 --num-classes 10 --model imixer_s16_224 --batch-size 512 --workers 4 --opt adamw --epochs 300 --sched cosine --amp --img-size 224 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20 --h_ratio 2 --n_power 8 --n_iter 1
```

## Reference

You may want to cite:

```
@article{ota2023imixer,
  title={iMixer: hierarchical Hopfield network implies an invertible, implicit and iterative MLP-Mixer},
  author={Ota, Toshihiro and Taki, Masato},
  year={2023}
}
```

## Acknowledgment

Our implementation is based on [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) by Ross Wightman. We greatly appreciate his brilliant work.

|   |   |
|:--|:-:|
|  We thank [Graduate School of Artificial Intelligence and Science, Rikkyo University (Rikkyo AI)](https://ai.rikkyo.ac.jp) which supports us with computational resources, facilities, and others. |  ![logo-rikkyo-ai] |

[logo-rikkyo-ai]: img/RIKKYOAI_main.png "Logo of Rikkyo AI"
