import pandas as pd
import time
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset, DatasetDict, load_dataset
from tqdm.auto import tqdm
import numpy as np
import torch

tqdm.pandas()

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

    for lan in langs:
        model = SetFitModel.from_pretrained("paraphrase-MiniLM-L3-v2", multi_target_strategy="multi-output",
                                            device='cuda')
        trainer = SetFitTrainer(
            model=model,
            train_dataset=ds[f'{lan}_train'],
            column_mapping={"combo": "text", "labels": "label"},
            num_epochs=5 if lan == 'java' else 10,
            batch_size=32,
        )
        trainer.train()
        trainer.model.save_pretrained(f'./models/{lan}')

    total_flops = 0
    total_time = 0
    scores = []
    for lan in langs:
        # to load trained models:
        model = SetFitModel.from_pretrained(f'./models/{lan}')
        # to load pretrained models from Hub:
        # model = SetFitModel.from_pretrained(f"NLBSE/nlbse25_{lan}")
        with torch.profiler.profile(with_flops=True) as p:
            begin = time.time()
            for i in range(10):
                y_pred = model(ds[f'{lan}_test']['combo']).numpy().T
            total = time.time() - begin
            total_time = total_time + total
        total_flops = total_flops + (sum(k.flops for k in p.key_averages()) / 1e9)
        y_true = np.array(ds[f'{lan}_test']['labels']).T
        for i in range(len(y_pred)):
            assert (len(y_pred[i]) == len(y_true[i]))
            tp = sum([true == pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
            tn = sum([true == pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
            fp = sum([true == 0 and pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
            fn = sum([true == 1 and pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2 * tp) / (2 * tp + fp + fn)
            scores.append({'lan': lan, 'cat': labels[lan][i], 'precision': precision, 'recall': recall, 'f1': f1})
    print("Compute in GFLOPs:", total_flops / 10)
    print("Avg runtime in seconds:", total_time / 10)
    scores = pd.DataFrame(scores)
    print(scores)

    max_avg_runtime = 5
    max_avg_flops = 5000


    # s𝑢𝑏𝑚𝑖𝑠𝑠𝑖𝑜𝑛_𝑠𝑐𝑜𝑟𝑒(𝑚𝑜𝑑𝑒𝑙)=(𝑎𝑣𝑔. 𝐹1)×0.60+((𝑚𝑎𝑥_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒−𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒)/𝑚𝑎𝑥_𝑎𝑣𝑔_𝑟𝑢𝑛𝑡𝑖𝑚𝑒)×0.2+((𝑚𝑎𝑥_GFLOPs−𝑚𝑒𝑎𝑠𝑢𝑟𝑒𝑑_GFLOPs)/𝑚𝑎𝑥_GFLOPs)×0.2
    def score(avg_f1, avg_runtime, avg_flops):
        return (0.6 * avg_f1 +
                0.2 * ((max_avg_runtime - avg_runtime) / max_avg_runtime) +
                0.2 * ((max_avg_flops - avg_flops) / max_avg_flops))


    avg_f1 = scores.f1.mean()
    avg_runtime = total_time / 10
    avg_flops = total_flops / 10

    round(score(avg_f1, avg_runtime, avg_flops), 2)