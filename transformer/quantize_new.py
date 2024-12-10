from transformers import RobertaForSequenceClassification, RobertaTokenizer, BitsAndBytesConfig
import numpy as np
import random
import torch
import os
from datasets import load_dataset, Dataset
from main import read_args, langs, labels as labels_langs

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")

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


from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from datasets import load_dataset


def preprocess_fn(ex, tokenizer):
    return tokenizer(ex['combo'], padding="max_length", truncation=True)



if __name__ == "__main__":

    args = read_args()
    lan = "java"
    onnx_model = ORTModelForSequenceClassification.from_pretrained(args.input_path, export=True, library="transformers")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)

    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')


    # Load the training dataset
    calibration_dataset = quantizer.get_calibration_dataset(
        ds[f"{lan}_train"],
        preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
        num_samples=200,
        dataset_split="train",
    )

    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)

    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )

    model_quantized_path = quantizer.quantize(
        save_dir=os.path.join(args.input_path, "q_2_"+lan, "models"),
        calibration_tensors_range=ranges,
        quantization_config=qconfig,
    )