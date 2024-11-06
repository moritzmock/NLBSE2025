import argparse
import os.path
import shutil
import time
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from datasets import load_dataset, Dataset
import re
from itertools import product
import numpy as np
import torch


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


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)


def modify_data(data):
    data = Dataset.from_dict({"text": data["combo"], "labels": data["labels"]})
    data = data.map(tokenize, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data = data.remove_columns(['text'])
    data = data.map(lambda x: {"labels": x["labels"].float()})

    return data


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # Convert labels to float if needed
        if labels is not None:
            inputs["labels"] = labels.float()
        outputs = model(**inputs)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


def create_path_structure(path, clear):
    if not os.path.exists(path):
        path_parts = path.split("/")
        current_path = ""

        for part in path_parts:
            current_path = os.path.join(current_path, part)
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


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--clear-output-path", default=True, type=str2bool)
    parser.add_argument("--hs", default=True, type=str2bool, help="If true, the hyperparameters are overwritten")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--weight-decay", default=0.01)
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--eval-strategy", default="no", choices=["no", "epoch"])

    return parser.parse_args()


def train_models(args, ds):
    print(args)

    for lan in langs:
        print(f"Training the model for the language {lan}...")

        create_path_structure(f"{args.output_path}/{lan}/results", args.clear_output_path)
        create_path_structure(f"{args.output_path}/{lan}/log", args.clear_output_path)
        create_path_structure(f"{args.output_path}/{lan}/models", args.clear_output_path)

        train = ds[f"{lan}_train"]
        test = ds[f"{lan}_test"]

        model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=len(labels[lan]))
        model.config.problem_type = "multi_label_classification"

        print("Model was loaded successfully!")

        train_data_complete = modify_data(train)
        test_data = modify_data(test)
        train_data = train_data_complete.train_test_split(test_size=len(test_data)/len(train_data_complete), seed=42)["train"] if args.eval_strategy != "no" else train_data_complete
        val_data = train_data_complete.train_test_split(test_size=len(test_data)/len(train_data_complete), seed=42)["test"] if args.eval_strategy != "no" else None

        print("Dataset was mutated successfully!")

        training_args = TrainingArguments(
            output_dir=f"{args.output_path}/{lan}/results",
            eval_strategy=args.eval_strategy,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            logging_dir=f"{args.output_path}/{lan}/logs"
        )

        print("TrainingArguments were created successfully!")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        print("Trainer was created successfully!")

        trainer.train()

        trainer.save_model(f"{args.output_path}/{lan}/models")

        result = trainer.evaluate(eval_dataset=test_data)

        print(result)

        for i, key in enumerate(labels[lan]):
            result = rename_keys_with_regex(result, f"eval_class_{i}", f"eval_{lan}_class_{key}")
            result = rename_keys_with_regex(result, f"eval_class", f"eval_{lan}_class")

        job_id = os.getenv("SLURM_JOB_ID")
        path = os.path.join(args.output_path, f"all_results_{job_id}.csv")
        csv_data = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

        index = len(csv_data) if langs.index(lan) == 0 else len(csv_data) - 1

        csv_data.loc[index, "info"] = generate_information(args, job_id)

        for key in result.keys():
            print(f"{key}: {result[key]}")
            csv_data.loc[index, key] = result[key]

        print(csv_data.keys())

        csv_data.to_csv(path, index=False)

        print("---------------------")



def convert_duration(seconds):
    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute

    # Format the result as dd-hh-mm-ss
    return f"{days:02d}-{hours:02d}-{minutes:02d}-{seconds:02d}"


if __name__ == "__main__":
    args = read_args()
    print(args)

    langs = ['python'] #['java', 'python', 'pharo']
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }

    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')
    print("Dataset was loaded successfully!")

    tokenizer = RobertaTokenizer.from_pretrained(args.model)

    print("Tokenizer was loaded successfully!")

    start_time = time.time()

    if args.hs == False:
        train_models(args, ds)

    if args.hs == True:
        epochs = [1, 3, 5, 10]
        lr = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        eval_strategy = ["no", "epoch"]
        batch_size = [1, 2, 4, 8, 16, 32]
        weight_decay = [0, 0.01, 0.001]
        arrays = [epochs, lr, eval_strategy, batch_size, weight_decay]
        combinations = generate_combinations(*arrays)

        for index, combination in enumerate(combinations):
            args.epochs = combination[0]         # overwrites the parameter
            args.lr = combination[1]             # overwrites the parameter
            args.eval_strategy = combination[2]  # overwrites the parameter
            args.batch_size = combination[3]     # overwrites the parameter
            args.weight_decay = combination[4]   # overwrites the parameter

            print(f"Execution number {index+1} out of {len(combinations)}")

            train_models(args, ds)

    end_time = time.time()

    duration = end_time - start_time
    print(f"Execution time: {convert_duration(duration)}")