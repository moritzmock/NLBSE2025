import pandas as pd
from utils import read_args, get_baseline
import math


def extract_information(
    data = None,
    global_average = -math.inf,
    global_index = 0,
    global_result = [],
    average_baseline = -math.inf,
    relevant_keys_precision = [],
    relevant_keys_recall = [],
    relevant_keys_f1 = []
):
    assert len(relevant_keys_f1) == len(relevant_keys_recall) == len(relevant_keys_precision)

    for index, row in data.iterrows():

        result = []

        for key in relevant_keys_f1:
            value = row[key]
            result.append(value)

        average = sum(result)/len(relevant_keys_f1) * 100
        if average > global_average:
            global_average = average
            global_index = index
            global_result = result
            if args.print_logs == "yes":
                print("new best value found")
                print(result)
                print("Diff to base line",  average_baseline - average)
                print("Baseline average", average_baseline)
                print("Average found", average)
                print("Index", index)
                print("Info", row["info"])
                print(args)

                print("\n\n-------------\n\n")

    print("Results")

    for key in relevant_keys_f1:
        position = relevant_keys_f1.index(key)
        p, r, f1 = data.loc[global_index, relevant_keys_precision[position]], \
            data.loc[global_index, relevant_keys_recall[position]], \
            data.loc[global_index, relevant_keys_f1[position]]

        print(f"{key:<50} - {round(p*100, 1):<15} { round(r*100, 1):<15} { round(f1*100, 1):<15}")
        if len(relevant_keys_f1) == 19 and (position == 6 or position == 11):
            print("------------------")

    return global_result


if __name__ == "__main__":
    args = read_args()
    print(args)

    data = pd.read_csv(args.input_path)

    baseline = get_baseline()

    relevant_keys_f1 = [key for key in data.keys().tolist() if key.endswith("_f1")]
    relevant_keys_precision = [key for key in data.keys().tolist() if key.endswith("_precision")]
    relevant_keys_recall = [key for key in data.keys().tolist() if key.endswith("_recall")]




    print("Best average result, there is no difference between the models for the paramters")
    average_baseline = baseline[:19]["F1"].sum() / 19
    r = extract_information(
        data=data,
        average_baseline=average_baseline,
        relevant_keys_f1=relevant_keys_f1,
        relevant_keys_precision=relevant_keys_precision,
        relevant_keys_recall=relevant_keys_recall
    )

    print("Best F1 for the same hyperparamters")
    print(sum(r) / len(r))
    print(len(r))

    print("Best individual hyperparamter combination foro JAVA")
    average_baseline = baseline[:7]["F1"].sum() / 7
    r_java = extract_information(
        data=data,
        average_baseline=average_baseline,
        relevant_keys_f1=relevant_keys_f1[:7],
        relevant_keys_precision=relevant_keys_precision[:7],
        relevant_keys_recall=relevant_keys_recall[:7]
    )
    print("Best F1 for different hyperparamters - java")
    print(sum(r_java) / len(r_java))
    print(len(r_java))


    print("Best individual hyperparamter combination foro Python")
    average_baseline = baseline[7:12]["F1"].sum() / 5
    r_python = extract_information(
        data=data,
        average_baseline=average_baseline,
        relevant_keys_f1=relevant_keys_f1[7:12],
        relevant_keys_precision=relevant_keys_precision[7:12],
        relevant_keys_recall=relevant_keys_recall[7:12]
    )
    print("Best F1 for different hyperparamters - python")
    print(sum(r_python) / len(r_python))
    print(len(r_python))

    print("Best individual hyperparamter combination foro Pharo")
    average_baseline = baseline[12:19]["F1"].sum() / 7
    r_pharo = extract_information(
        data=data,
        average_baseline=average_baseline,
        relevant_keys_f1=relevant_keys_f1[12:19],
        relevant_keys_precision=relevant_keys_precision[12:19],
        relevant_keys_recall=relevant_keys_recall[12:19]
    )
    print("Best F1 for different hyperparamters - pharo")
    print(sum(r_pharo) / len(r_pharo))
    print(len(r_pharo))

    r = r_java + r_python + r_pharo

    print("Best F1 for different hyperparamters")
    print(sum(r)/len(r))
    print(len(r))