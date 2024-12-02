from transformers import Trainer
import torch, torch.nn as nn
import numpy as np
from MTL.utils import compute_metrics


class CustomTrainer(Trainer):
            def __init__(self, *args, weight_method, weight_method_name, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_method = weight_method
                self.weight_method_name = weight_method_name
                self.global_step = 0
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # Extract labels and logits
                if "labels" not in inputs:
                    raise ValueError("Labels are missing in the inputs.")
                
                labels = inputs.pop("labels")
                logits = model(**inputs).logits

                # Ensure labels are float
                if labels.dtype != torch.float:
                    labels = labels.float()
                # Compute loss
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                loss = loss_fct(logits, labels)
                
                return (loss, logits) if return_outputs else loss
            
            def evaluate(self, eval_dataset=None, **kwargs):
                """
                Custom evaluation loop to handle predictions and labels more carefully.
                """
                print("Starting evaluation...")
                self.model.eval()  # Ensure the model is in eval mode
                
                # Use DataLoader to iterate over the eval dataset
                eval_dataloader = self.get_eval_dataloader(eval_dataset)
                
                all_logits = []
                all_labels = []
                
                # Evaluate in batches
                for batch in eval_dataloader:
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}
                     # Forward pass
                    with torch.no_grad():
                        batch["labels"] = batch["labels"].float() 
                        model_output = self.model(**batch)  # Model output can vary in structure
                        
                        # If the model returns a tuple, access logits using the appropriate index
                        if isinstance(model_output, tuple):
                            logits = model_output[0]  # Assuming the first element is logits
                        elif isinstance(model_output, dict):
                            logits = model_output.get("logits")  # For ModelOutput, access logits via key
                        else:
                            raise ValueError(f"Unexpected model output format: {type(model_output)}")

                    
                    # Collect predictions and labels
                    labels = batch["labels"]
                    all_logits.append(logits)
                    all_labels.append(labels)
                
                all_logits = torch.cat(all_logits, dim=0)  # Concatenate and convert to numpy
                all_labels = torch.cat(all_labels, dim=0)
                
                # Handle edge cases where predictions may be empty
                if all_logits.shape[0] == 0:
                    raise ValueError("No predictions available for evaluation.")
                
                # Compute metrics
                metrics = self.compute_metrics((all_logits,all_labels))
                print(f"Evaluation metrics: {metrics}")
                
                return metrics
            def training_step(self, model, inputs):
                """
                Perform a custom training step.
                """
                # Move inputs to device
                inputs_d = {k: v.to(self.args.device) for k, v in inputs.items()}
                # labels = inputs.pop("labels")  # Assuming 'labels' is part of the dataset

                # # Forward pass
                # self.optimizer.zero_grad()
                # model.train()
                # outputs = model(**inputs)
                # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                
                # if labels.dtype != torch.float:
                #    labels=labels.float()
                
                losses = self.compute_loss(model,inputs_d) #loss_fct(outputs.get("logits").squeeze(), labels.squeeze()) #
                
                if losses.dim() > 1:  # Check if batch dim s greater than 1
                    per_sample_loss = losses.mean(dim=0)  # Compute mean across heads for each sample
                else:
                    per_sample_loss = losses
                # Apply custom backward method with weighted losses
                loss, extra_outputs = self.weight_method.backward(
                    losses=per_sample_loss
                )
                
                # Print every 500 steps
                self.global_step += 1
                if self.global_step % 500 == 0:
                    print(f"\nStep {self.global_step}: Collected Extra Outputs:")
                    for key, value in extra_outputs.items():
                        print(f"{key}: {value}")

                # Perform optimizer step
                self.optimizer.step()

                if "famo" in self.weight_method_name:
                    with torch.no_grad():
                        # Forward pass again for updated losses
                        # self.compute_loss(model,labels,inputs)
                        # train_pred = model(**inputs)
                        new_losses = self.compute_loss(model,inputs)#loss_fct(train_pred.get("logits").squeeze(), labels.squeeze())
                        if losses.dim() > 1:  # Check if batch dim s greater than 1
                            per_sample_new_losses = new_losses.mean(dim=0)  # Compute mean across heads for each sample
                        else:
                            per_sample_new_losses = new_losses
                        
                        self.weight_method.method.update(per_sample_new_losses.detach())

                return per_sample_loss.sum()