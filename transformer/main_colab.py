from transformers import RobertaForSequenceClassification, RobertaTokenizer
import numpy as np
import random
import torch
import os
from datasets import load_dataset, Dataset
import time
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


if __name__ == "__main__":
    langs = ['java', 'python', 'pharo']
    labels_langs = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }

    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')

    set_seed(42)

    # for all the models codebert is performing the best, therefore, we only need to load its tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    total_flops = 0
    total_time = 0
    scores = []

    for lan in langs:
        test = ds[f"{lan}_test"]

        model = RobertaForSequenceClassification.from_pretrained(f"mmock/NLBSE2025_{lan}")
        model.to(device)
        model.eval()

        predictions = []

        test_data = test.to_pandas()

        for index, row in test_data.iterrows():
            combo = row["combo"]
            input_ids = tokenizer(combo, return_tensors="pt").to(device)
            start_time = time.time()
            with torch.profiler.profile(with_flops=True) as p:
                # Forward pass
                for i in range(10):
                    outputs = model(**input_ids)
            end_time = time.time()
            total_time = total_time + end_time - start_time
            total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)
            # Extract logits (raw outputs)
            logits = outputs.logits

            # Apply sigmoid activation to get probabilities
            probs = torch.sigmoid(logits)

            # Convert probabilities to binary predictions (threshold = 0.5)
            threshold = 0.5
            predictions.append((probs > threshold).int().cpu().numpy().tolist()[0])

        labels = torch.tensor(test_data["labels"].tolist()).to(device)
        labels = labels.cpu().numpy()

        num_classes = len(labels_langs[lan])
        class_metrics = {}

        predictions = np.array(predictions)
        labels = np.array(labels)

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
            class_metrics[f"class_{labels_langs[lan][class_idx]}_true_positives"] = tp
            class_metrics[f"class_{labels_langs[lan][class_idx]}_false_positives"] = fp
            class_metrics[f"class_{labels_langs[lan][class_idx]}_true_negatives"] = tn
            class_metrics[f"class_{labels_langs[lan][class_idx]}_false_negatives"] = fn
            class_metrics[f"class_{labels_langs[lan][class_idx]}_precision"] = precision
            class_metrics[f"class_{labels_langs[lan][class_idx]}_recall"] = recall
            class_metrics[f"class_{labels_langs[lan][class_idx]}_f1"] = f1
            scores.append(
                {'lan': lan, 'cat': labels_langs[lan][class_idx], 'precision': precision, 'recall': recall, 'f1': f1})

        print(f"results for {lan}")

        for key in class_metrics.keys():
            print(key, class_metrics[key])

    print("----------------RESULTS--------------------")

    print("Compute in GFLOPs:", total_flops / 10)
    print("Avg runtime in seconds:", total_time / 10)
    scores = pd.DataFrame(scores)
    scores




