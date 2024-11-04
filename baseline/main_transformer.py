import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from evaluate import load

accuracy = load("accuracy")
precision = load("precision")
recall = load("recall")
f1 = load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)


if __name__ == "__main__":
    langs = ['java', 'python', 'pharo']
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }

    lan = langs[0]

    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')
    print("Dataset was loaded successfully!")

    train = ds[f"{lan}_train"]
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(labels[lan]))
    model.config.problem_type = "multi_label_classification"


    print("Model was loaded successfully!")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("Tokenizer was loaded successfully!")

    train_data = Dataset.from_dict({"text": train["combo"], "labels": train["labels"]})\
        .map(tokenize, batched=True)\
        .set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print("Dataset was mutated successfully!")

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="no",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    print("TrainingArguments were created successfully!")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        compute_metrics=compute_metrics,
    )

    print("Trainer was created successfully!")

    trainer.train()