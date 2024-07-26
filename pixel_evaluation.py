from tqdm import tqdm
import numpy as np

from PIL import Image
from collections import Counter
from multiprocessing import Pool
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc


def counts_array_to_data_list(counts_array, max_size=None):
    if max_size is None:
        max_size = np.sum(counts_array)  # max of counted array entry
    if np.sum(counts_array) != 0: 
        counts_array = (counts_array / np.sum(counts_array) * max_size).astype("uint32")
    counts_dict = {}
    for i in range(1, len(counts_array) + 1):
        counts_dict[i] = counts_array[i - 1]
    return list(Counter(counts_dict).elements())


def calc_precision_recall(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+5)
    else:
        ratio_in = np.sum(data["in"]) / (np.sum(data["in"]) + np.sum(data["out"]))
        ratio_out = 1 - ratio_in
        x1 = counts_array_to_data_list(np.array(data["in"]), 1e+7 * ratio_in)
        x2 = counts_array_to_data_list(np.array(data["out"]), 1e+7 * ratio_out)
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2))))
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    return precision_recall_curve(y_true, y_scores) + (average_precision_score(y_true, y_scores), )


def calc_sensitivity_specificity(data, balance=False):
    if balance:
        x1 = counts_array_to_data_list(np.array(data["in"]), max_size=1e+5)
        x2 = counts_array_to_data_list(np.array(data["out"]), max_size=1e+5)
    else:
        x1 = counts_array_to_data_list(np.array(data["in"]))
        x2 = counts_array_to_data_list(np.array(data["out"]))
    probas_pred1 = np.array(x1) / 100
    probas_pred2 = np.array(x2) / 100
    y_true = np.concatenate((np.zeros(len(probas_pred1)), np.ones(len(probas_pred2)))).astype("uint8")
    y_scores = np.concatenate((probas_pred1, probas_pred2))
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds, auc(fpr, tpr)

def pixel_wise_metrics_i(ood_heat_load_path, gt_load_path):
    ood_heat = np.load(ood_heat_load_path)
    gt = np.array(Image.open(gt_load_path))
    

    return pixel_metrics(ood_heat, gt)


def pixel_metrics(ood_heat, gt, num_bins=100, in_label=0, out_label=254):
    bins = np.linspace(start=0, stop=1, num=num_bins + 1)
    counts = {"in": np.zeros(num_bins, dtype="int32"), "out": np.zeros(num_bins, dtype="int32")}
    counts["in"] += np.histogram(ood_heat[gt == in_label], bins=bins, density=False)[0]
    counts["out"] += np.histogram(ood_heat[gt == out_label], bins=bins, density=False)[0]
    return counts



def aggregate_pixel_metrics(frame_results: list):
    print("Could take a moment...")
    num_bins = len(frame_results[0]["out"])
    counts = {"in": np.zeros(num_bins, dtype="int64"), "out": np.zeros(num_bins, dtype="int64")}
    for r in frame_results:
        counts["in"] += r["in"]
        counts["out"] += r["out"]
    fpr, tpr, _, auroc = calc_sensitivity_specificity(counts, balance=True)
    fpr95 = fpr[(np.abs(tpr - 0.95)).argmin()]
    _, _, _, auprc = calc_precision_recall(counts)
    print("AUROC : {:6.2f} %".format(auroc*100))
    print("FPR95 : {:6.2f} %".format(fpr95*100))
    print("AUPRC : {:6.2f} %".format(auprc*100))



def pixel_wise_evaluation(dataset, num_cpus=16):
    print("evaluate OOD scores")
    results = dataset.images 
    results = [f'{file.replace("_raw_data.jpg",".npy").replace("_raw_data.png",".npy").replace("raw_data", "ood_score")}'for file in dataset.images]
    
    target_paths = dataset.targets_semantic_ood
    targets = list(filter(lambda ele:ele is not None, target_paths))

    pool_args = [(results[i], targets[i]) for i in range(len(results))]
        
    with Pool(num_cpus) as pool:
        results = pool.starmap(pixel_wise_metrics_i, tqdm(pool_args, total=pool_args.__len__()), chunksize = 4)
    aggregate_pixel_metrics(results)

