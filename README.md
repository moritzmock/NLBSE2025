# NLBSE2025

This repository contains the replication package for the work *Optimizing Deep Learning Models to Address Class Imbalance in Code Comment Classification*. The work was conducted by Moritz Mock, Thomas Borsani, Giuseppe Di Fatta, and Barbara Russo.

Link to preprint: [https://doi.org/10.48550/arXiv.2501.15854](https://doi.org/10.48550/arXiv.2501.15854)

## Abstract

Developers rely on code comments to document their work, track issues, and understand the source code. As such, comments provide valuable insights into developers' understanding of their code and describe their various intentions in writing the surrounding code. Recent research leverages natural language processing and deep learning to classify comments based on developers' intentions. However, such labelled data are often imbalanced, causing learning models to perform poorly.
This work investigates the use of different weighting strategies of the loss function to mitigate the scarcity of certain classes in the dataset. In particular, various RoBERTa-based transformer models are fine-tuned by means of a hyperparameter search to identify their optimal parameter configurations. Additionally, we fine-tuned the transformers with different weighting strategies for the loss function to address class imbalances.
Our approach outperforms the STACC baseline by 8.9 per cent on the NLBSE'25 Tool Competition dataset in terms of the average F1$_c$ score, and exceeding the baseline approach in 17 out of 19 cases with a gain ranging from -5.0 to 38.2.
 The source code is publicly available at https://github.com/moritzmock/NLBSE2025.

## Installation

For the weighted loss (*ranked* and *frequency*) and no weighted loss follow the instructions below:

```
cd transformer
python -m venv env
source env/bin/activate
pip install -r req.txt
```

While for FAMO use the following instructions:

```
cd NLBASE2025
python -m venv env
source env/bin/activate
pip install -r req.txt
```

## Running the scripts

For the transformer there are two main options how to run the model, using the --hs flag such that the other parameters are ignored, e.g., not using it such that the parameters can be passed:

```
python main.py --output-path <path> --model-path <model> --hs True
```

For the FAMO is it possible to run the model specifying the output-path as minimum parameter:

```
export PYTHONPATH=$(pwd)
python MTL/main.py --output-path ./results  --hs True --weight-method-name 'famo'
```

### Result of Hyper-Parameter Search

In the following table the best hyper-parameters which resulted into the best performing fine-tuned pre-trained models can be found. The values of the can be found at the paper (see above for the link).

| Dataset | Batch Size | Epochs | Learning Rate | Weight Decay | loss weights*  |
|---------|------------|--------|---------------|--------------|----------------|
| Java    | 4          | 10     | 3e-5          | 0.001        | RBF            |
| Python  | 2          | 10     | 4e-5          | 0            | RBF            |
| Pharo   | 4          | 10     | 3e-5          | 0.01         | ICF            |

While the *loss weights** is not part of a Hyper-Parameter Search in the traditional sense, we have considered to place it here as it influences the performance of a pre-trained model in the fine-tuning step.

## Availability of the models
The best performing models are available at huggingface:

Java: https://huggingface.co/mmock/NLBSE2025_java

Pharo: https://huggingface.co/mmock/NLBSE2025_pharo

Python: https://huggingface.co/mmock/NLBSE2025_python

## Google Colab

In the following the notebook which was used for the inference and the calculation of the submission score can be found: https://colab.research.google.com/drive/1q6x-x5MwhgnRkjErBpbvxMSZAE_UypuO?usp=sharing

The results which we have obtained are the following: F1 72.6, avg. runtime of 11.6 seconds, GFLOPs of 155,300, submission score 0.44. 

## How to cite the work

#### Preprint:

```bibtext
@misc{mock2025optimizingdeeplearningmodels,
      title={Optimizing Deep Learning Models to Address Class Imbalance in Code Comment Classification}, 
      author={Moritz Mock and Thomas Borsani and Giuseppe Di Fatta and Barbara Russo},
      year={2025},
      eprint={2501.15854},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2501.15854}, 
}
```
