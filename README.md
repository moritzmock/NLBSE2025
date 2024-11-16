# NLBSE2025
This repository contains code for solving a multi-task learning (MTL) problem using transformers, with a focus on sequence classification tasks. Each label corresponds to a separate task, and the code adapts the problem to handle dynamic weighting systems alongside static weightings.

## MTL folder
The MTL folder contains the core code for the multi-task problem setup. Each file plays a specific role in implementing the multi-task model and ensuring its flexibility for various use cases.

- **`utils.py`**  
This file contains all utility functions used in the original code. No modifications have been made to these utilities. They are directly imported and used as-is.

- **`trainer.py`**
This is a custom trainer based on the Hugging Face (HF) transformers library. The trainer.py file modifies the original training loop to support a dynamic weighting system, while still providing the option to use static weightings. The dynamic weighting adjusts based on task performance during training.

- **`model.py`**
This file defines a custom model for the Hugging Face RobertaForSequenceClassification class.

In the standard RobertaForSequenceClassification, the classification head is defined as:
```
self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
```

In our custom model, we modify the final layer to support a multi-head, multi-task structure. This modification allows the model to perform classification tasks for multiple labels simultaneously, making it suitable for handling multi-task learning setups.

- **`weight_methods.py`**
    The repository includes two main weighting strategies for handling the loss functions in multi-task learning:

    - **FAMO (Dynamic Weighting Strategy)**  
    FAMO is a dynamic weighting strategy that adjusts the weight of each task's loss during training based on task performance. It allows the model to focus more on the tasks that are lagging behind or need more attention, making the training process more adaptive.

    - **LinearScalarization**  
    This strategy performs the sum of the losses from all tasks and applies a weighting vector to control the importance of each individual task. The weighting vector allows for static control over how each task contributes to the overall loss during training.

