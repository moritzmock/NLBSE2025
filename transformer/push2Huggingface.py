from main import read_args
from huggingface_hub import Repository
import logging


if __name__ == "__main__":
    args = read_args()

    logging.basicConfig(level=logging.DEBUG)

    model_name = "mmock/test"

    repo = Repository(local_dir=args.input_path, clone_from=model_name)

    repo.push_to_hub()

