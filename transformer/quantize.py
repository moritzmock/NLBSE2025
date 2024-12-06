from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import BitsAndBytesConfig
import numpy as np
import random
import torch
import os
from datasets import load_dataset, Dataset
from main import read_args, langs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()

# Print details about each device
print("Available devices:")
for i in range(num_gpus):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# Check if CUDA is available
if num_gpus == 0:
    print("No CUDA devices available. Defaulting to CPU.")
else:
    print(f"Number of GPUs available: {num_gpus}")

# Check the current device
print(f"Current device: {torch.cuda.current_device()}")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


def modify_data(data):
    data = Dataset.from_dict({"text": data["combo"], "labels": data["labels"]})
    data = data.map(tokenize, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data = data.remove_columns(['text'])
    data = data.map(lambda x: {key: val.to(device) for key, val in x.items()})
    # data = data.map(lambda x: {"labels": x["labels"].float()})

    return data

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


if __name__ == "__main__":
    set_seed(42)

    args = read_args()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    lan = langs[0]

    tokenizer = RobertaTokenizer.from_pretrained(args.model)
    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')

    test = ds[f"{lan}_test"]

    model = RobertaForSequenceClassification.from_pretrained(os.path.join(args.input_path, lan, "models"), quantization_config=nf4_config)

    print("model loaded...")

    print(test)

    test_data = modify_data(test)

    test_data = test_data.to_pandas()

    print(test_data)

