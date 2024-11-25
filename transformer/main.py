import argparse
import os.path
import random
import shutil
import time
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from datasets import load_dataset, Dataset
import re
from itertools import product
import numpy as np
import torch, torch.nn as nn
import math

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
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)


def modify_data(data):
    data = Dataset.from_dict({"text": data["combo"], "labels": data["labels"]})
    data = data.map(tokenize, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data = data.remove_columns(['text'])
    data = data.map(lambda x: {key: val.to(device) for key, val in x.items()})
    # data = data.map(lambda x: {"labels": x["labels"].float()})

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


class WeightedBCELoss(nn.Module):
    def __init__(self, weights, weighted_loss):
        super(WeightedBCELoss, self).__init__()
        self.weighted_loss = weighted_loss
        self.weights = torch.tensor(weights, device=device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels):
        logits = logits.to(device)
        labels = labels.to(device)

        loss = self.bce(logits, labels)  # Calculate unweighted BCE loss

        weighted_loss = None

        if self.weighted_loss == "yes": # naive implementation
            weighted_loss = np.array(loss) / math.log10(len(self.weights))
        else:
            weighted_loss = loss * self.weights  # Apply weights to each label's loss

        return weighted_loss.mean()


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
    parser.add_argument("--old-run", default="False", type=str,
                        help="Passing the csv file of an old run allows to skip executed experiments")
    parser.add_argument("--jobID-manual", default=None, required=False,
                        help="If the script is not executed in a SLURM environment, the job ID can be passed "
                             "to keep manually track of the multiple executions at the same time")
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--epochs", default=3)
    parser.add_argument("--weight-decay", default=0.01)
    parser.add_argument("--lr", default=5e-5)
    parser.add_argument("--eval-strategy", default="no", choices=["no", "epoch"])
    parser.add_argument("--weighted-loss", default="no", choices=["no", "yes"])

    return parser.parse_args()


def generate_weights(weighted_loss, data):

    '''

    if weighted_loss == "yes":
        labels = data["labels"]
        labels_array = np.array(labels)
        class_counts = labels_array.sum(axis=0)
        total_samples = len(labels)
        class_frequencies = (class_counts/total_samples)*100
        result = []

        for i, freq in enumerate(class_frequencies):
            result.append(round(1/freq, 2))

        result = np.array(result)

        return result * (1 / np.sum(result))

    '''


    # NO WEIGHTED LOSS
    return [1] * len(data[0]["labels"])


def train_models(args, ds, job_id):
    print(args)

    for lan in langs:
        print(f"Training the model for the language {lan}...")

        create_path_structure(f"{args.output_path}/{job_id}/{lan}/results", args.clear_output_path)
        create_path_structure(f"{args.output_path}/{job_id}/{lan}/log", args.clear_output_path)
        create_path_structure(f"{args.output_path}/{job_id}/{lan}/models", args.clear_output_path)

        train = ds[f"{lan}_train"]
        test = ds[f"{lan}_test"]

        label_weights = torch.tensor(generate_weights(args.weighted_loss, train))

        custom_loss = WeightedBCELoss(weights=label_weights, weighted_loss=args.weighted_loss)

        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels").float().to(device)  # Move labels to the global device
                inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to global device

                #labels = inputs.pop("labels").float()
                outputs = model(**inputs)
                logits = outputs.logits
                loss = custom_loss(logits, labels)
                return (loss, outputs) if return_outputs else loss

        model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=len(labels[lan]))
        model.config.problem_type = "multi_label_classification"
        model = model.to(device)

        print("Model was loaded successfully!")

        train_data_complete = modify_data(train)
        test_data = modify_data(test)
        train_data = train_data_complete.train_test_split(test_size=len(test_data)/len(train_data_complete), seed=42)["train"] if args.eval_strategy != "no" else train_data_complete
        val_data = train_data_complete.train_test_split(test_size=len(test_data)/len(train_data_complete), seed=42)["test"] if args.eval_strategy != "no" else None

        print("Dataset was mutated successfully!")

        training_args = TrainingArguments(
            output_dir=f"{args.output_path}/{job_id}/{lan}/results",
            eval_strategy=args.eval_strategy,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            learning_rate=args.lr,
            logging_dir=f"{args.output_path}/{job_id}/{lan}/logs"
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

        trainer.save_model(f"{args.output_path}/{job_id}/{lan}/models")

        result = trainer.evaluate(eval_dataset=test_data)

        result = rename_keys_with_regex(result, "eval_", "eval_" + lan + "_")
        result = rename_keys_with_regex(result, "epoch", "epoch_" + lan)
        for i, key in enumerate(labels[lan]):
            result = rename_keys_with_regex(result, "eval_" + lan + "_class_" + str(i), "eval_" + lan + "_class_" + key)

        path = os.path.join(args.output_path, f"all_results_{job_id}.csv")
        csv_data = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

        index = len(csv_data) if langs.index(lan) == 0 else len(csv_data) - 1

        csv_data.loc[index, "info"] = generate_information(args, job_id)

        for key in result.keys():
            print(f"{key}: {result[key]}")
            csv_data.loc[index, key] = result[key]

        csv_data.to_csv(path, index=False)

        print("---------------------")



def convert_duration(seconds):
    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute

    # Format the result as dd-hh-mm-ss
    return f"{days}-{hours}-{minutes}-{seconds}"


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
    set_seed(42)
    args = read_args()
    print(args)

    langs = ['java', 'python', 'pharo']
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

    job_id = os.getenv("SLURM_JOB_ID") if os.getenv("SLURM_JOB_ID") is not None else args.jobID_manual

    assert job_id is not None, \
        f"The script is not executed in an SLURM environment and/or no '--jobID-manual' parameter was passed"

    if args.hs == False:
        train_models(args, ds, job_id)

    if args.hs == True:
        epochs = [1, 3, 5, 10]
        lr = [3e-5, 4e-5, 5e-5] # 1e-5, 2e-5, 
        eval_strategy = ["no"]
        batch_size = [1, 2, 4, 8, 16]
        weight_decay = [0, 0.01, 0.001]
        arrays = [epochs, lr, eval_strategy, batch_size, weight_decay]
        combinations = generate_combinations(*arrays)

        print(args.old_run)

        path = os.path.join(args.output_path, args.old_run) if args.old_run != "False" else None
        print(path)
        prior_executions = pd.read_csv(path) if path is not None else None

        if prior_executions is not None:
            assert prior_executions["info"].tolist()[0].split("_")[1] == args.model, \
            f"Expected model '{args.model}', but got '{prior_executions['info'].tolist()[0].split('_')[1]}'." \
            f" Most likely passed the wrong old run file"
            # removing the last experiment, since it might not be fully completed
            prior_executions = prior_executions[:-1]

            info_list = prior_executions["info"].tolist()
            info_list = [tuple(execution.split("_")[2:]) for execution in info_list]

            for index, execution in enumerate(info_list):
                info_list[index] = (int(execution[1]), float(execution[3]), execution[4], int(execution[0]), float(execution[2]) if execution[2] != "0" else int(execution[2]))

            c_old_len = len(combinations)

            combinations = [item for item in combinations if item not in info_list]

            assert c_old_len - len(combinations) == len(prior_executions), "Skipping of experiments was not calculated " \
                                                                           "correctly..."

            path = os.path.join(args.output_path, f"all_results_{job_id}.csv")
            prior_executions.to_csv(path, index=False)

            print(f"Skipped {len(prior_executions)} number of experiments!")

        for index, combination in enumerate(combinations):
            args.epochs = combination[0]         # overwrites the parameter
            args.lr = combination[1]             # overwrites the parameter
            args.eval_strategy = combination[2]  # overwrites the parameter
            args.batch_size = combination[3]     # overwrites the parameter
            args.weight_decay = combination[4]   # overwrites the parameter

            print(f"Execution number {index+1} out of {len(combinations)}")

            train_models(args, ds, job_id)

    end_time = time.time()

    duration = end_time - start_time
    print(f"Execution time: {convert_duration(duration)}")