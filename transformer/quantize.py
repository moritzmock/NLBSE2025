import argparse

from transformers import RobertaForSequenceClassification
from transformers import BitsAndBytesConfig
import numpy as np
import random
import torch

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


def read_args():

    args = argparse.ArgumentParser()

    args.add_argument("--input-path", required=True)

    return args.parse_args()


if __name__ == "__main__":
    set_seed(42)

    args = read_args()

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = RobertaForSequenceClassification.from_pretrained(args.input_path, quantization_config=nf4_config)

