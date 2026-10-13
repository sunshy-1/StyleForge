<div align="center">
<img src="fig/logo.png" alt="logo" width="2200" style="display:block; margin-bottom:0;"/>
<img src="https://img.shields.io/badge/version-1.0.1-6395ED" alt="version"/>
<img src="https://img.shields.io/badge/license-MIT-9ACD32" alt="license"/>
<a href="https://arxiv.org/abs/2610.05066"><img src="https://img.shields.io/badge/Preprint'26-EE4C2C" alt="preprint"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"/></a>
<img src="https://img.shields.io/github/stars/sunshy-1/JuDi?style=social" alt="stars"/>
</div>

<br/>



This is the Pytorch implementation for our paper: [**Salvation Lies Within: Eliciting Inherent Style Transfer in Step-Distilled Diffusion Models**](https://arxiv.org/abs/2610.05066). 

## Abstract
<div style="text-align: justify;">
  Adapting step-distilled text-to-image (T2I) models through post-training incurs additional computational costs and affects native few-step generation behavior. This motivates a complementary route beyond style-specific adaptation: drawing on the visual knowledge already encoded in step-distilled T2I models to elicit stylistic capabilities through language. Pursuing this direction requires textual guidance that captures how visual attributes jointly define a style and remain applicable as the depicted content changes. To explore this approach, we introduce StyleForge, a fully automatic, training-free framework that expresses reference styles as reusable rendering instructions. By integrating overall rendering characteristics with local color and lighting behavior, StyleForge organizes visual evidence from reference images into a coherent specification of how the target style should be expressed. The specification is then compiled into textual guidance that can be reused across content prompts, enabling frozen step-distilled T2I models to render different subjects and scenes in the reference style while retaining native few-step generation. Extensive experiments show relative gains of up to 29.47\% in generation quality scores over the strongest baseline, while Pareto analysis indicates that improved stylization is accompanied by strong adherence to the requested content.
</div>
<br>

![Framework](fig/framework.png)

## Requirements

The StyleForge generation pipeline uses Python 3.10 and the following key packages:

```shell
PyTorch==2.10.0
Transformers==5.10.1
Diffusers==0.38.0
Accelerate==1.14.0
NumPy==1.26.4
```

Evaluation uses a separate Python 3.10 environment with SGLang to run the
vision-language model judges. Its key packages are:

```shell
SGLang==0.5.10.post1
PyTorch==2.9.1
Transformers==5.3.0
Triton==3.5.1
NumPy==1.26.4
```

Both environments include CUDA 12.8 runtime packages. For complete package
versions, see the two environment snapshots in [requirements.txt](requirements.txt).
These snapshots are not a `pip install -r` requirements file.
