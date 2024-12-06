from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer, RobertaTokenizer, DataCollatorWithPadding
from transformers import BitsAndBytesConfig
import numpy as np
import random
import torch
import os
from datasets import load_dataset, Dataset
import pandas as pd
from main import generate_weights, WeightedBCELoss, read_args, modify_data, rename_keys_with_regex, labels, langs, generate_information

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

    train = ds[f"{lan}_train"]
    test = ds[f"{lan}_test"]

    model = RobertaForSequenceClassification.from_pretrained(os.path.join(args.input_path, lan, "models"), quantization_config=nf4_config)

    print("model loaded...")

    label_weights = torch.tensor(generate_weights(args.weighted_loss, train))

    custom_loss = WeightedBCELoss(weights=label_weights)


    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels").float().to(device)  # Move labels to the global device
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to global device

            # labels = inputs.pop("labels").float()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = custom_loss(logits, labels)
            return (loss, outputs) if return_outputs else loss

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    job_id = args.input_path.split("/")[-1]

    training_args = TrainingArguments(
        output_dir=f"{args.output_path}/{job_id}/{lan}/q_results",
        eval_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        logging_dir=f"{args.output_path}/{job_id}/{lan}/q_logs"
    )

    train_data_complete = modify_data(train)
    test_data = modify_data(test)
    train_data = train_data_complete.train_test_split(test_size=len(test_data) / len(train_data_complete), seed=42)[
        "train"] if args.eval_strategy != "no" else train_data_complete
    val_data = train_data_complete.train_test_split(test_size=len(test_data) / len(train_data_complete), seed=42)[
        "test"] if args.eval_strategy != "no" else None

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.save_model(f"{args.output_path}/{job_id}/{lan}/q_models")

    result = trainer.evaluate(eval_dataset=test_data)

    result = rename_keys_with_regex(result, "eval_", "eval_" + lan + "_")
    result = rename_keys_with_regex(result, "epoch", "epoch_" + lan)
    for i, key in enumerate(labels[lan]):
        result = rename_keys_with_regex(result, "eval_" + lan + "_class_" + str(i), "eval_" + lan + "_class_" + key)

    path = os.path.join(args.output_path, f"all_results_{job_id}_q.csv")
    csv_data = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    index = len(csv_data) if langs.index(lan) == 0 else len(csv_data) - 1

    csv_data.loc[index, "info"] = generate_information(args, job_id)

    for key in result.keys():
        print(f"{key}: {result[key]}")
        csv_data.loc[index, key] = result[key]

    csv_data.to_csv(path, index=False)

    print("---------------------")

