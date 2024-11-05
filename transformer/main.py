import argparse
import os.path
import shutil
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    print("Predictions shape:", predictions.shape)
    print("Labels shape:", labels.shape)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")

    # Calculate TP, FP, TN, FN
    TP = np.sum((predictions == 1) & (labels == 1), axis=0)
    FP = np.sum((predictions == 1) & (labels == 0), axis=0)
    TN = np.sum((predictions == 0) & (labels == 0), axis=0)
    FN = np.sum((predictions == 0) & (labels == 1), axis=0)

    # Aggregate results across all classes
    tp_total = TP.sum()
    fp_total = FP.sum()
    tn_total = TN.sum()
    fn_total = FN.sum()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "tn": tn_total,
        "fn": fn_total,
    }




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


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--clear-output-path", default=True, type=str2bool)

    return parser.parse_args()


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


if __name__ == "__main__":
    args = read_args()
    print(args)

    #langs = ['java', 'python', 'pharo']
    langs = ['python'] # todo remove
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }

    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')
    print("Dataset was loaded successfully!")

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

        tokenizer = RobertaTokenizer.from_pretrained(args.model)

        print("Tokenizer was loaded successfully!")

        train_data = modify_data(train)
        test_data = modify_data(test)

        print("Dataset was mutated successfully!")

        training_args = TrainingArguments(
            output_dir=f"{args.output_path}/{lan}/results",
            eval_strategy="no",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f"{args.output_path}/{lan}/logs"
        )

        print("TrainingArguments were created successfully!")

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        print("Trainer was created successfully!")

        trainer.train()

        trainer.save_model(f"{args.output_path}/{lan}/models")

        result = trainer.evaluate(eval_dataset=test_data)

        print(result)