from transformers import Trainer
import torch, torch.nn as nn

class CustomTrainer(Trainer):
            def __init__(self, *args, weight_method, weight_method_name, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_method = weight_method
                self.weight_method_name = weight_method_name

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
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                
                if labels.dtype != torch.float:
                   labels=labels.float()
                losses = loss_fct(outputs.get("logits").squeeze(), labels.squeeze())

                # Apply custom backward method with weighted losses
                total_loss, extra_outputs = self.weight_method.backward(
                    losses=losses
                )

                # Perform optimizer step
                self.optimizer.step()

                if "famo" in self.weight_method_name:
                    with torch.no_grad():
                        # Forward pass again for updated losses
                        train_pred = model(**inputs)
                        new_losses = loss_fct(train_pred.get("logits").squeeze(), labels.squeeze())
                        self.weight_method.method.update(new_losses.detach())

                return total_loss