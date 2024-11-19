import argparse
import os.path
import shutil
from collections import defaultdict

import re
from itertools import product
import numpy as np
import torch
import random
from transformers import set_seed

def generate_combinations(*arrays):
    return list(product(*arrays))


def compute_metrics(eval_pred):
    # Unpack the predictions and labels
    logits, labels = eval_pred

    # Convert logits to a PyTorch tensor if they are in NumPy format
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Apply sigmoid to logits to get probabilities (for multilabel)
    probabilities = torch.sigmoid(logits)  # Shape: (num_samples, num_classes)

    # Convert probabilities to binary predictions based on a threshold
    threshold = 0.5
    predictions = (probabilities >= threshold).int()  # Binarize predictions

    # Ensure predictions and labels are on the same device (CPU)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # Check if predictions and labels are compatible
    num_classes = labels.shape[1]
    class_metrics = {}

    # Calculate TP, FP, TN, FN per class
    for class_idx in range(num_classes):
        tp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 1))
        fp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 0))
        tn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 0))
        fn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 1))

        # Calculate precision, recall, and F1 score for the current class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics for the current class
        class_metrics[f"class_{class_idx}_true_positives"] = tp
        class_metrics[f"class_{class_idx}_false_positives"] = fp
        class_metrics[f"class_{class_idx}_true_negatives"] = tn
        class_metrics[f"class_{class_idx}_false_negatives"] = fn
        class_metrics[f"class_{class_idx}_precision"] = precision
        class_metrics[f"class_{class_idx}_recall"] = recall
        class_metrics[f"class_{class_idx}_f1"] = f1

    return class_metrics

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def create_path_structure(path, clear):
    if not os.path.exists(path):
        path_parts = path.split("/")
        current_path = ""

        for part in path_parts:
            current_path = os.path.join(current_path, part)
            print(current_path)
            if not os.path.exists(current_path) and current_path != "":
                
                os.makedirs(current_path)
                print(f"Directory created: {current_path}")
            else:
                print(f"Directory already exists: {current_path}")
            if current_path == "":
                current_path = "/"

        print("Path structure created successfully.")

    if clear is True:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)

            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
                print(f"File removed: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and all its contents
                print(f"Directory removed: {item_path}")

        print(f"All objects in {path} have been removed.")


def rename_keys_with_regex(d, old_prefix, new_prefix):
    pattern = re.compile(r'^' + re.escape(old_prefix))

    new_dict = {}
    for key in d:
        if pattern.match(key):
            new_key = pattern.sub(new_prefix, key)
            new_dict[new_key] = d[key]
        else:
            new_dict[key] = d[key]

    return new_dict


def generate_information(args, jobID):
    return f"{jobID}_" \
           f"{args.model}_" \
           f"{args.batch_size}_" \
           f"{args.epochs}_" \
           f"{args.weight_decay}_" \
           f"{args.lr}_" \
           f"{args.eval_strategy}"


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            famo=dict(gamma=args.gamma,
                      w_lr=args.method_params_lr,
                      max_norm=args.max_norm),
        )
    )
    return weight_methods_parameters