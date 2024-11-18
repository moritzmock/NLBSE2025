from transformers import Trainer
import torch, torch.nn as nn
# test
class CustomTrainer(Trainer):
            def __init__(self, *args, weight_method, weight_method_name, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_method = weight_method
                self.weight_method_name = weight_method_name
                self.global_step = 0

            def training_step(self, model, inputs):
                """
                Perform a custom training step.
                """
                # Move inputs to device
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                labels = inputs.pop("labels")  # Assuming 'labels' is part of the dataset

                # Forward pass
                self.optimizer.zero_grad()
                model.train()
                outputs = model(**inputs)
                loss_fct = nn.BCEWithLogitsLoss()
                
                if labels.dtype != torch.float:
                   labels=labels.float()

                losses = loss_fct(outputs.get("logits").squeeze(), labels.squeeze())
                if losses.dim() > 1:  # Check if batch dim s greater than 1
                    per_sample_loss = losses.mean(dim=1)  # Compute mean across heads for each sample
                else:
                    per_sample_loss = losses
                
                # Apply custom backward method with weighted losses
                total_loss, extra_outputs = self.weight_method.backward(
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
                        train_pred = model(**inputs)
                        new_losses = loss_fct(train_pred.get("logits").squeeze(), labels.squeeze())
                        if losses.dim() > 1:  # Check if batch dim s greater than 1
                            per_sample_new_losses = new_losses.mean(dim=1)  # Compute mean across heads for each sample
                        else:
                            per_sample_new_losses = new_losses
                        
                        self.weight_method.method.update(per_sample_new_losses.detach())

                return total_loss