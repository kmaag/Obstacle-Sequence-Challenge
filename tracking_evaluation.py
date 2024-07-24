import os
import time
import pickle
import concurrent.futures
import numpy as np
from PIL import Image
from pathlib import Path


def comp_remaining_metrics(tracking_metrics):
    """
    helper function for metrics calculation
    """

    tracking_metrics["mot_a"] = 1 - (
        (
            tracking_metrics["fn"]
            + tracking_metrics["fp"]
            + tracking_metrics["switch_id"]
        )
        / (tracking_metrics["tp"] + tracking_metrics["fn"])
    )

    tracking_metrics["mismatches"] = tracking_metrics["switch_id"] / (
        tracking_metrics["tp"] + tracking_metrics["fn"]
    )

    tracking_metrics["precision"] = tracking_metrics["tp"] / (
        tracking_metrics["tp"] + tracking_metrics["fp"]
    )

    tracking_metrics["recall"] = tracking_metrics["tp"] / (
        tracking_metrics["tp"] + tracking_metrics["fn"]
    )

    tracking_metrics["mot_p"] = tracking_metrics["center_dist"] / tracking_metrics["tp"]

    return tracking_metrics

def merge_results(io_root, sequences):
    print("merge results over sequences")

    tracking_metrics = {
        "num_frames": np.zeros((1)),
        "gt_objects": np.zeros((5, 2)),
        "tp": np.zeros((1)),
        "fp": np.zeros((1)),
        "fn": np.zeros((1)),
        "precision": np.zeros((1)),
        "recall": np.zeros((1)),
        "matching": list([]),
        "tracking_length": np.zeros((2)),
        "mot_a": np.zeros((1)),
        "switch_id": np.zeros((1)),
        "mismatches": np.zeros((1)),
        "mot_p": np.zeros((1)),
        "center_dist": np.zeros((1)),
        "GT": np.zeros((1)),
        "mostly_tracked": np.zeros((1)),
        "partially_tracked": np.zeros((1)),
        "mostly_lost": np.zeros((1)),
    }

    with open(Path(io_root) / f"tracking_eval" / "results_table.txt", "wt") as fi:
        for seq in sequences:
            print(seq)
            

            name = "results_" + seq + ".p"
            folder_path = Path(io_root) / f"tracking_eval" / name
            
            if not folder_path.exists():
                continue
            
            results = pickle.load(open(Path(io_root) / f"tracking_eval" / name, "rb"))

            tracking_results = comp_remaining_metrics(results)
            print(seq, tracking_results, file=fi)
            print(" ", file=fi)

            for tm in tracking_metrics:
                if tm in [
                    "num_frames",
                    "gt_objects",
                    "tp",
                    "fp",
                    "fn",
                    "tracking_length",
                    "switch_id",
                    "center_dist",
                    "GT",
                    "mostly_tracked",
                    "partially_tracked",
                    "mostly_lost",
                ]:
                    tracking_metrics[tm] += results[tm]

        tracking_results = comp_remaining_metrics(tracking_metrics)
        print(tracking_results, file=fi)
        return tracking_results

def eval_tracking_seq(seq, io_root, basenames, img_paths):
    print("start", seq)
    start = time.time()

    gt_max = 5
    results = {
        "num_frames": np.zeros((1)),
        "gt_objects": np.zeros((gt_max, 2)),
        "tp": np.zeros((1)),
        "fp": np.zeros((1)),
        "fn": np.zeros((1)),
        "precision": np.zeros((1)),
        "recall": np.zeros((1)),
        "matching": list([]),
        "tracking_length": np.zeros((2)),
        "mot_a": np.zeros((1)),
        "switch_id": np.zeros((1)),
        "mismatches": np.zeros((1)),
        "mot_p": np.zeros((1)),
        "center_dist": np.zeros((1)),
        "GT": np.zeros((1)),
        "mostly_tracked": np.zeros((1)),
        "partially_tracked": np.zeros((1)),
        "mostly_lost": np.zeros((1)),
    }

    for idx in range(len(basenames)):
        if seq in basenames[idx]:
            break

    target_paths = []
    num_imgs = 0
    for i in range(len(basenames)):
        target_paths.append(None)
        if seq in basenames[i]:
            num_imgs += 1
            target_path = (
                img_paths[i].replace("raw_data", "instance_ood").replace("jpg", "png")
            )
            if os.path.isfile(target_path):
                target_paths[-1] = target_path
    print("seq/index/num img/num gt: ", seq, idx, num_imgs, results["GT"])

    tp_tmp = np.zeros((gt_max))
    matching = {}
    for j in range(gt_max):
        matching["lengths_" + str(j)] = list([])
        matching["ids_" + str(j)] = list([])

    for n in range(num_imgs):
        l_n = n + idx
        print(seq, n, basenames[l_n])

        results["num_frames"] += 1

        components = np.load(
            Path(io_root) / f"ood_prediction_tracked" / f"{basenames[l_n]}.npy"
        )
        if target_paths[l_n] != None:
            gt = np.array(Image.open(target_paths[l_n]))

            gt_unique = np.unique(gt)
            results["GT"] = max(results["GT"], len(gt_unique) - 1)
            for j in range(gt_max):
                if j + 1 in gt_unique:
                    results["gt_objects"][j, 0] += 1
                    results["gt_objects"][j, 1] += 1

            comp_unique = np.unique(components)
            results["fp"] += len(comp_unique) - 1

            # gt ids - max iou, corresponding seg id
            iou_id = np.ones((gt_max, 2)) * -1

            for i in comp_unique[1:]:
                for j in range(gt_max):
                    intersection = np.sum(
                        np.logical_and(gt == (j + 1), components == i)
                    )
                    union = np.sum(np.logical_or(gt == (j + 1), components == i))
                    if union > 0 and intersection > 0:
                        if intersection / union > iou_id[j, 0]:
                            iou_id[j, 0] = intersection / union
                            iou_id[j, 1] = i

            match_ids = np.unique(iou_id[:, 1])
            if match_ids[0] == -1:
                match_ids = match_ids[1:]
            results["fp"] -= len(match_ids)

            for j in range(gt_max):
                if iou_id[j, 0] > 0:
                    tp_tmp[j] += 1
                    x_y_component = np.sum(
                        np.asarray(np.where(components == iou_id[j, 1])), axis=1
                    )
                    x_y_gt = np.sum(np.asarray(np.where(gt == (j + 1))), axis=1)
                    results["center_dist"] += (
                        (
                            x_y_component[1] / np.sum(components == iou_id[j, 1])
                            - x_y_gt[1] / np.sum(gt == (j + 1))
                        )
                        ** 2
                        + (
                            x_y_component[0] / np.sum(components == iou_id[j, 1])
                            - x_y_gt[0] / np.sum(gt == (j + 1))
                        )
                        ** 2
                    ) ** 0.5
                elif j + 1 in gt_unique:
                    results["fn"] += 1

                # new match with pred segment for a gt segment
                if (len(matching["ids_" + str(j)]) == 0 and iou_id[j, 0] > 0) or (
                    len(matching["ids_" + str(j)]) != 0
                    and iou_id[j, 0] > 0
                    and iou_id[j, 1] != matching["ids_" + str(j)][-1]
                ):
                    matching["ids_" + str(j)].append(iou_id[j, 1])
                    matching["lengths_" + str(j)].append(0)
                # increase tracking lengths
                if (
                    len(matching["ids_" + str(j)]) != 0
                    and iou_id[j, 1] == matching["ids_" + str(j)][-1]
                ):
                    matching["lengths_" + str(j)][-1] += 1

        else:
            for j in range(gt_max):
                if results["gt_objects"][j, 0] > 0:
                    results["gt_objects"][j, 0] += 1

                if (
                    len(matching["ids_" + str(j)]) != 0
                    and np.sum(components == matching["ids_" + str(j)][-1]) > 0
                ):
                    matching["lengths_" + str(j)][-1] += 1

    for j in range(gt_max):
        if len(matching["ids_" + str(j)]) > 0:
            results["switch_id"] += len(matching["ids_" + str(j)]) - 1

    for j in range(gt_max):
        results["tracking_length"][0] += np.sum(matching["lengths_" + str(j)])
    results["tracking_length"][1] = np.sum(results["gt_objects"][:, 0])

    results["tp"] += np.sum(tp_tmp)
    for j in range(gt_max):
        if results["gt_objects"][j, 1] > 0:
            quotient = tp_tmp[j] / results["gt_objects"][j, 1]
            if quotient >= 0.8:
                results["mostly_tracked"] += 1
            elif quotient >= 0.2:
                results["partially_tracked"] += 1
            else:
                results["mostly_lost"] += 1

    results["matching"] = matching

    print("results", results)

    name = "results_" + seq + ".p"
    pickle.dump(results, open(Path(io_root) / f"tracking_eval" / name, "wb"))
    print("sequence", seq, "processed in {}s\r".format(round(time.time() - start, 4)))



def evaluate_tracking(dataset, io_root, num_cpus=1):
    if not os.path.exists(Path(io_root) / f"tracking_eval"):
        os.makedirs(Path(io_root) / f"tracking_eval")

    if num_cpus == 1:
        for seq in dataset.sequences:
            eval_tracking_seq(
                seq, io_root, dataset.basenames, dataset.images    
            )
    else:
        p_args = [
            (seq, io_root, dataset.basenames, dataset.images)
            for seq in dataset.sequences
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executer:
            executer.map(eval_tracking_seq, *zip(*p_args))

    tracking_metrics = merge_results(io_root, dataset.sequences)
    return tracking_metrics