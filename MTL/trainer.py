from transformers import Trainer
import torch, torch.nn as nn

class CustomTrainer(Trainer):
            def __init__(self, *args, weight_method, weight_method_name, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_method = weight_method
                self.weight_method_name = weight_method_name
                self.global_step = 0
            
            def compute_loss(self, model, inputs, return_outputs=False):
                # Extract labels and logits
                
                labels = inputs.pop("labels")

                logits = model(**inputs).logits

                # Ensure labels are float
                if labels.dtype != torch.float:
                    labels = labels.float()

                # Compute loss
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                loss = loss_fct(logits, labels)

                return (loss, logits) if return_outputs else loss

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

                losses = self.compute_loss(model,inputs_d) #loss_fct(outputs.get("logits").squeeze(), labels.squeeze())
                
                if losses.dim() > 1:  # Check if batch dim s greater than 1
                    per_sample_loss = losses.mean(dim=0)  # Compute mean across heads for each sample
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
                        # self.compute_loss(model,labels,inputs)
                        # train_pred = model(**inputs)
                        new_losses = self.compute_loss(model,inputs)#loss_fct(train_pred.get("logits").squeeze(), labels.squeeze())
                        if losses.dim() > 1:  # Check if batch dim s greater than 1
                            per_sample_new_losses = new_losses.mean(dim=0)  # Compute mean across heads for each sample
                        else:
                            per_sample_new_losses = new_losses
                        
                        self.weight_method.method.update(per_sample_new_losses.detach())

                return total_loss