# Sequence-to-function deep learning frameworks for engineered riboregulators

This repository provides code for [Valeri, Collins, Ramesh, et al. 2020](https://www.biorxiv.org/content/10.1101/870055v1). 

## Summary
We introduce STORM and NuSpeak, two deep learning pipelines that work in concert to characterize and optimize synthetic riboregulators.

## Abstract
While synthetic biology has revolutionized our approaches to medicine, agriculture, and energy, the design of completely novel biological circuit components beyond naturally-derived templates remains challenging due to poorly understood design rules. Toehold switches, which are programmable nucleic acid sensors, face an analogous design bottleneck; our limited understanding of how sequence impacts functionality often necessitates expensive and time-consuming screens to identify effective switches. Here, we introduce Sequence-based Toehold Optimization and Redesign Model (STORM) and Nucleic-Acid Speech (NuSpeak), two orthogonal and synergistic deep learning architectures to characterize and optimize toeholds. Applying techniques from computer vision and natural language processing, we ‘un-box’ both models using convolutional filters, attention maps, and in silico mutagenesis. Through transfer-learning, we redesign sub-optimal toehold sensors, even with sparse training data (N = 168), and we experimentally validate their improved performance. Our deep learning platform provides sequence-to-function frameworks for toehold selection and design, augmenting our ability to construct potent biological circuit components and precision diagnostics.

## Analysis
In the clean_figures/ folder, we have code to reproduce each figure in the paper. There are also two notebooks in the main folder corresponding to STORM's two purposes: prediction and redesign. First, a notebook to use the trained model for predicting ON and OFF values of toehold sequences has been uploaded. Second, a notebook to use the trained model and our gradient ascent framework for optimizing toehold sequences has been uploaded. Example sequences for both are located in their respective folders, where the output of the notebook will display. Notebooks to reproduce all figures are being made available, with some language model notebooks pending.

## Website
A web version of these tools has been made available to ease integration into lab workflows. The beta version of our website is available at https://storm-toehold.herokuapp.com. Please note there is a ~10 second delay on startup if the website has not been used in a while. For any feedback, questions, or bug reports, email valerij@mit.edu.

## Running notebooks
This virtual environment and packages have only been tested on a Mac running Mojave, so no guarantees if you have another system or OS.
    
0. Make a virtual environment with conda and python 3.7 (assume both are already installed)
```
    conda create -n myenv python=3.7 anaconda
    conda activate myenv
```
    
1. Install git-lfs and git clone the repository.
```
    pip3 install git-lfs
    git clone https://github.com/midas-wyss/engineered-riboregulator-ML
```    

2. Download everything in the requirements.txt package:
```
    pip3 install -r requirements.txt
```
    
3. Run jupyter notebook (once in the notebook, make sure to switch the KERNEL to myenv- drop down Kernel menu and click Change kernnel) after adding the new venv to your list of jupyter kernels with ipykernel.
```
    pip3 install --user ipykernel
    python -m ipykernel install --user --name=myenv
    jupyter notebook
```
    
4. To leave the venv, run:
```
    conda deactivate
```
