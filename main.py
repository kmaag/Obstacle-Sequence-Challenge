import argparse
from datasets.street_obstacle_sequences import StreetObstacles
from datasets.wuppertal_obstacle_sequences import WuppertalObstacles
from pixel_evaluation import pixel_wise_evaluation
from segment_evaluation import segment_wise_evaluation
from tracking_evaluation import evaluate_tracking
    
def main(args):
    if args.dataset == 'sos':
        dataset = StreetObstacles(args.dataset_path)
    elif args.dataset == 'wos':
        dataset = WuppertalObstacles(args.dataset_path)
    
    pixel_wise_evaluation(dataset)
    segment_wise_evaluation(dataset)
    evaluate_tracking(dataset,args.dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OoD Tracking Evaluation")
    parser.add_argument(
        "--dataset_path",
        metavar="FILE",
        help="path to the dataset folder to be evaluated"
    )

    parser.add_argument("--dataset", choices=['sos', 'wos'], help="Choose the dataset: 'sos' or 'cwl'")
    args = parser.parse_args()
    main(args)
    print("Done")