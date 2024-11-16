import argparse
import os.path
import shutil
import time
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Dataset
import re
from itertools import product
import numpy as np
import torch, torch.nn as nn

from MTL.utils import create_path_structure,rename_keys_with_regex,generate_information,str2bool,compute_metrics,generate_combinations
from MTL.weight_methods import WeightMethods
from MTL.model import RobertaForSequenceMultiLabelClassification


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

def modify_data(data):
    data = Dataset.from_dict({"text": data["combo"], "labels": data["labels"]})
    data = data.map(tokenize, batched=True)
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    data = data.remove_columns(['text'])
    data = data.map(lambda x: {"labels": x["labels"].float()})

    return data






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
    parser.add_argument("--weighted-loss", default="famo", choices=["ls", "wls","famo"])


    return parser.parse_args()


def generate_weights(weighted_loss, data):

    if weighted_loss == "wls":
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

        # custom_loss = WeightedBCELoss(weights=label_weights, weighted_loss=args.weighted_loss)
        # loss_fn   = nn.BCEWithLogitsLoss("none")

        # weight method
        weight_method = WeightMethods(
            args.weighted_loss, n_tasks=len(labels[lan]) ,device='cpu' #TODO set this to work generally
        )

        # class FAMOWeightedTrainer(Trainer):
        #     def __init__(self, *args, loss_weight_manager, **kwargs):
        #         super().__init__(*args, **kwargs)
        #         self.loss_weight_manager = loss_weight_manager  # Instance of your loss weighting class

        #     def compute_loss(self, model, inputs, return_outputs=False):
        #         # Forward pass
        #         outputs = model(**inputs)
                
        #         losses = outputs.get("losses")  # List of losses for each task
        #         if not isinstance(losses, list):
        #             raise ValueError("Model must return a list of losses.")

        #         # Get dynamic weights from the manager
        #         task_weights = self.loss_weight_manager.get_weights()

        #         # Apply weights to the losses
        #         weighted_loss = 0.0
        #         for i, loss in enumerate(losses):
        #             weight = task_weights.get(i, 1.0)  # Default weight is 1.0
        #             weighted_loss += weight * loss

        #         return (weighted_loss, outputs) if return_outputs else weighted_loss

        #     def training_step(self, model, inputs):
        #         # Perform the regular training step
        #         loss = super().training_step(model, inputs)

        #         # Update the weights dynamically after the step
        #         self.loss_weight_manager.update()

        #         return loss
            

        class CustomTrainer(Trainer):
            def __init__(self, *args, weight_method, alpha, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_method = weight_method
                self.alpha = alpha

            def training_step(self, model, inputs):
                """
                Perform a custom training step.
                """
                # Move inputs to device
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                labels = inputs.pop("labels")  # Assuming 'labels' is part of the dataset

                # Forward pass
                model.train()
                outputs = model(**inputs)
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                
                if labels.dtype != torch.float:
                   labels=labels.float()
                print(f"training step logist:{outputs.get("logits").shape} ,labels: {labels.shape}")
                losses = loss_fct(outputs.get("logits").squeeze(), labels.squeeze())
                print(f"training step outputs: {losses}")

                # Apply custom backward method with weighted losses
                total_loss, extra_outputs = self.weight_method.backward(
                    losses=losses * self.alpha
                )

                # Perform optimizer step
                self.optimizer.zero_grad()
                # total_loss.backward()
                self.optimizer.step()

                # Update weight method if required
                with torch.no_grad():
                    # Forward pass again for updated losses
                    train_pred = model(**inputs)
                    new_losses = loss_fct(train_pred.get("logits").squeeze(), labels.squeeze())
                    self.weight_method.method.update(new_losses.detach())

                return total_loss
        
        print('set model')
        model = RobertaForSequenceMultiLabelClassification.from_pretrained(args.model, num_labels=len(labels[lan]))
        model.config.problem_type = "multi_label_classification"

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
            weight_method=weight_method,
            alpha=0.2,
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


if __name__ == "__main__":
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

    job_id = 1 #os.getenv("SLURM_JOB_ID") if os.getenv("SLURM_JOB_ID") is not None else args.jobID_manual

    assert job_id is not None, \
        f"The script is not executed in an SLURM environment and/or no '--jobID-manual' parameter was passed"

    if args.hs == False:
        train_models(args, ds, job_id)

    if args.hs == True:
        epochs = [1, 3, 5, 10]
        lr = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        eval_strategy = ["no", "epoch"]
        batch_size = [1, 2, 4, 8, 16, 32]
        weight_decay = [0, 0.01, 0.001]
        arrays = [epochs, lr, eval_strategy, batch_size, weight_decay]
        combinations = generate_combinations(*arrays)

        path = os.path.join(args.output_path, args.old_run) if args.old_run != "False" else None
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