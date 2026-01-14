# Differentiable Surrogate for Detector Simulation and Design with Diffusion Models

This repository contains the code for the paper:

**"Differentiable Surrogate for Detector Simulation and Design with Diffusion Models"**  
by Xuan Tung Nguyen et al., https://doi.org/10.48550/arXiv.2601.07859

The project provides a conditional denoising-diffusion probabilistic model (DDPM) to simulate electromagnetic calorimeter showers. The model can generate high-fidelity, differentiable shower distributions conditioned on detector geometry, material, and incoming particle energy.



![Calo shower](https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design/blob/main/images/compare_XY4_Z10_small.png)
![Energy pprofile](https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design/blob/main/images/energy_profiles_PbF2_xy4_z10_small.png)


## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation
```bash
git clone https://github.com/X-T-Nguyen/Diffusion-Surrogate-Detector-Design.git
cd Diffusion-Surrogate-Detector-Design
conda create -n diff-surrogate python=3.11
conda activate diff-surrogate
pip install -r requirements.txt
```

## Data
The dataset used in this work is publicly available on Zenodo: https://doi.org/10.5281/zenodo.17105137

## Training
Pre-training:
```bash
python MainCondition.py
```

Post-training:
```bash
python fine_tune.py
```

## Evaluation:
The evaluation scripts are provided to assess model performance and generate key analysis outputs. These include:

<details>
<summary><b>Visual comparison between generated and ground-truth showers</b></summary>
  
```bash
python shower_plot.py
```
</details> 

<details>
<summary><b>Computation and plotting of longitudinal and transverse energy profiles</b></summary>
  
```bash
python edep_plot.py
```
</details> 


<details>
<summary><b>Evaluation of physical fidelity metrics</b></summary>
  
```bash
python metric_plot.py
```
</details> 

<details>
<summary><b>Gradient-based analysis comparing the foundation model and the post-trained model</b></summary>
  
```bash
python grad_plot.py
```
</details> 

## Citation
If you use this code in your research, please cite:

```bibtex
@article{nguyen2025diffsurrogate,
  title={Differentiable Surrogate for Detector Simulation and Design with Diffusion Models},
  author={Nguyen, Xuan Tung and Chen, Long and Dorigo, Tommaso and Gauger, Nicolas R. and others},
  year={2025},
  journal={tbd},

```

## Acknowledgments
This work was supported by the [MODE Collaboration](https://mode-collaboration.github.io/) and the Alliance for High Performance Computing in Rhineland-Palatinate (AHRP) via the Elwetritsch cluster at RPTU Kaiserslautern-Landau.

