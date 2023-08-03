# Sequence-to-function deep learning frameworks for engineered riboregulators

This repository provides code for [Valeri, Collins, Ramesh, et al. 2020](https://www.nature.com/articles/s41467-020-18676-2). 

## Summary
We introduce STORM and NuSpeak, two deep learning pipelines that work in concert to characterize and optimize synthetic riboregulators.

## Abstract
While synthetic biology has revolutionized our approaches to medicine, agriculture, and energy, the design of completely novel biological circuit components beyond naturally-derived templates remains challenging due to poorly understood design rules. Toehold switches, which are programmable nucleic acid sensors, face an analogous design bottleneck; our limited understanding of how sequence impacts functionality often necessitates expensive, time-consuming screens to identify effective switches. Here, we introduce Sequence-based Toehold Optimization and Redesign Model (STORM) and Nucleic-Acid Speech (NuSpeak), two orthogonal and synergistic deep learning architectures to characterize and optimize toeholds. Applying techniques from computer vision and natural language processing, we ‘un-box’ our models using convolutional filters, attention maps, and in silico mutagenesis. Through transfer-learning, we redesign sub-optimal toehold sensors, even with sparse training data, experimentally validating their improved performance. This work provides sequence-to-function deep learning frameworks for toehold selection and design, augmenting our ability to construct potent biological circuit components and precision diagnostics.

## Analysis
In the clean_figures/ folder, we have code to reproduce key figures and statistics from the manuscript. There are also example notebooks in the main folder corresponding to demos for NuSpeak and STORM. For the CNN-based predictions, a notebook to use the trained model for predicting ON and OFF values of toehold sequences has been uploaded as well as a notebook to use the trained model and our gradient ascent framework for optimizing toehold sequences has been uploaded. Example sequences for both are located in their respective folders, where the output of the notebook will display. Additionally, corresponding notebooks for the language model prediction and optimization are available. Please contact valerij@mit.edu for clarifications/comments/issues.

## Website
A web version of these tools has been made available to ease integration into lab workflows. The beta version of our website is available at https://storm-toehold.herokuapp.com. Please note there is a ~10 second delay on startup if the website has not been used in a while. If you have any issues, questions, or concerns, please do not hesitate to open up an issue or contact jackievaleri8 "at" gmail "dot" com.

## Running notebooks
This virtual environment and packages have only been tested on a Mac running Mojave. If you are running a different OS, some issues may arise.
    
0. Make a virtual environment with conda and python 3.7 (assume both are already installed)
```
    conda create -n myenv python=3.7 anaconda
    conda activate myenv
```
    
1. Install git-lfs and git clone the repository.
```
    brew install git-lfs
    git lfs install
    git clone https://github.com/midas-wyss/engineered-riboregulator-ML
    cd engineered-riboregulator-ML/
    git lfs checkout
    git lfs fetch
    git lfs pull
```    

2. Download everything in the requirements.txt package:
```
    pip3 install -r requirements.txt
```
    
3. Run jupyter notebook (once in the notebook, make sure to switch the KERNEL to myenv- drop down Kernel menu and click Change kernel) after adding the new venv to your list of jupyter kernels with ipykernel.
```
    python -m ipykernel install --user --name=myenv
    jupyter notebook
```
    
4. To leave the venv, run:
```
    conda deactivate
```
