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
@misc{nguyen2025diffsurrogate,
  title        = {Differentiable Surrogate for Detector Simulation and Design with Diffusion Models},
  author       = {Nguyen, Xuan Tung and Chen, Long and Dorigo, Tommaso and Gauger, Nicolas R. and Vischia, Pietro and Nardi, Federico and Awais, Muhammad and Hanif, Hamza and Abbas, Shahzaib and Kapoor, Rukshak},
  year         = {2025},
  eprint       = {2601.07859},
  archivePrefix= {arXiv},
  primaryClass = {physics.ins-det},
  doi          = {10.48550/arXiv.2601.07859}
}


```

## Acknowledgments
This work was carried out within the MODE Collaboration, and we thank its members for valuable discussions.

We acknowledge funding and computing support from the German National High Performance Computing (NHR) association (Center NHR South-West), the Alliance for High Performance Computing in Rhineland-Palatinate (AHRP) via the Elwetritsch cluster at RPTU Kaiserslautern-Landau, and the Artemisa computing infrastructure funded by the European Union ERDF and the Comunitat Valenciana.

Pietro Vischia was supported by the Ramón y Cajal programme (Project No. RYC2021-033305-I) funded by MCIN/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR.

We also acknowledge technical support from the Instituto de Física Corpuscular (IFIC, CSIC–UV).

