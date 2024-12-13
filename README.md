# NLBSE2025

This repository contains the replication package for the work *Optimizing Deep Learning Models to Address Class Imbalance in Code Comment Classification*. The work was conducted by Moritz Mock, Thomas Borsani, Giuseppe Di Fatta, and Barbara Russo.

Link to preprint: **under review**

## Abstract

Developers rely on code comments to document their work, track issues, and understand the source code. As such, comments provide valuable insights into developers' understanding of their code and describe their various intentions in writing the surrounding code. Recent research leverages natural language processing and deep learning to classify comments based on developers' intentions. Unfortunately, such labeled data are often imbalanced, causing learning models to perform poorly.
To address this, mitigating the scarcity of certain classes in the dataset is crucial.
In this study, we fine-tuned various RoBERTa-based transformer models and conducted a hyperparameter search to identify their optimal parameter configurations. Additionally, we executed the transformers with different loss functions to address class imbalances.
Our approach outperforms the STACC baseline by 8.7 per cent on the NLBSE’25 Tool Competition dataset in terms of average F1$_c$-score, also exceeding the baseline for 17 out of 19 cases with a gain ranging from -5.0 to 38.2.
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
TBD
```

## Running the scripts

For the transformer there are two main options how to run the model, using the --hs flag such that the other parameters are ignored, e.g., not not using it such that the parameters can be passed:

```
python main.py --output-path <path> --model-path <model> --hs True
```

## Availability of the models
The best performing models are available at huggingface:

Java: https://huggingface.co/mmock/NLBSE2025_java

Pharo: https://huggingface.co/mmock/NLBSE2025_pharo

Python:https://huggingface.co/mmock/NLBSE2025_python

## Google Colab

In the following there is an example implementation on how the models can be tested: https://colab.research.google.com/drive/1q6x-x5MwhgnRkjErBpbvxMSZAE_UypuO?authuser=1
