from main import read_args
from transformers import AutoModelForSequenceClassification
from huggingface_hub import HfApi, Repository

if __name__ == "__main__":
    args = read_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.input_path)

    model_name = "mmock/test"

    repo = Repository(local_dir=args.input_path, clone_from=model_name)

    repo.push_to_hub()

