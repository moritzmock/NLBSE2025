from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments

if __name__ == "__main__":
    langs = ['java', 'python', 'pharo']
    labels = {
        'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
        'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
        'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages',
                  'Collaborators']
    }

    # Load dataset
    ds = load_dataset('NLBSE/nlbse25-code-comment-classification')
    print("Dataset was loaded successfully!")

    lan = langs[0]

    model = SetFitModel.from_pretrained(
        "paraphrase-MiniLM-L3-v2",
        multi_target_strategy="multi-output",
    )
    print("Model was loaded successfully!")

    args = TrainingArguments(
        output_dir=f"/home/clusterusers/momock/NLBSE2025/results/baseline/reimplementation/models/{lan}",
        batch_size=32,
        num_epochs=5,
        save_steps=1,
        save_total_limit=2
    )
    print("TrainingArguments were set successfully!")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds[f'{lan}_train'].rename_column("combo", "text").rename_column("labels", "label"),
    )

    print("Trainer was set successfully!")

    trainer.train()

    trainer.model.save_pretrained(f'/home/clusterusers/momock/NLBSE2025/results/baseline/reimplementation/models/{lan}')
