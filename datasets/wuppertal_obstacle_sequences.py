import numpy as np
from pathlib import Path
from natsort import natsorted
from typing import Union, List, Any, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def read_image(path,label=False):
    if label:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
    else:   
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def convert_target(target_list, targets_basenames):
    target = []
    j = 0
    for i in range(len(targets_basenames)):
        if targets_basenames[i]:
            target.append(target_list[j])
            j += 1
        else:
            target.append(None)
    return target

class WuppertalObstacles:
    """
    Dataset class for the Wuppertal Obstacle Sequences dataset
    Args:
        root (string): Root directory of dataset
        sequences (string, optional): The image sequences to load
        target_type (string or list, optional): Type of target to use, choose from ("semantic_ood", "instance_ood").
        transforms (callable, optional): A function/transform that takes input sample and its target as entry and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        sequences: str = ["all"],
        target_type: Union[List[str], str] = "semantic_ood",
        transforms: Optional[Callable] = None,
    ):
        self.root = root
        self.images = []
        self.all_images = []
        self.targets_semantic_ood = []
        self.targets_instance_ood = []
        self.targets_semantic = []
        self.basenames = []
        self.all_basenames = []
        self.ood_id = 254
        self.target_type = target_type
        self.transforms = transforms
        self.ood_classes = np.arange(50, 57)
        self.id_dict = {
            "sequence_001": [1],
            "sequence_002": [1],
            "sequence_003": [1],
            "sequence_004": [1],
            "sequence_005": [1],
            "sequence_006": [1],
            "sequence_007": [1],
            "sequence_008": [1],
            "sequence_009": [1],
            "sequence_010": [1],
            "sequence_011": [1],
            "sequence_012": [1,2,3,4],
            "sequence_013": [1],
            "sequence_014": [1,2],
            "sequence_015": [1,2],
            "sequence_016": [1],
            "sequence_017": [1],
            "sequence_018": [1],
            "sequence_019": [1],
            "sequence_020": [1,2],
            "sequence_021": [1],
            "sequence_022": [1],
            "sequence_023": [1,2,3],
            "sequence_024": [1],
            "sequence_025": [1],
            "sequence_026": [1],
            "sequence_027": [1],
            "sequence_028": [1],
            "sequence_029": [1],
            "sequence_030": [1],
            "sequence_031": [1],
            "sequence_032": [1],
            "sequence_033": [1],
            "sequence_034": [1,2,3,4],
            "sequence_035": [1],
            "sequence_036": [1],
            "sequence_037": [1],
            "sequence_038": [1],
            "sequence_039": [1],
            "sequence_040": [1,2,3,4,5,6,7],
            "sequence_041": [2,3,4,5,6,7,8],
            "sequence_042": [1,2,3,4,5,6,7,8,9,10],
            "sequence_043": [1,2,4,5,7,8,9,10,11,12,13,14,15],
            "sequence_044": [1,2,3,4,5,6,7,8,9,12,13,14,15,16],

        }
        
        if not isinstance(target_type, list):
            self.target_type = [target_type]

        if sequences is None or "all" in [str(s).lower() for s in sequences]:
            self.sequences = []
            for sequence in (Path(self.root) / "raw_data").glob("sequence*"):
                self.sequences.append(str(sequence.name))
        elif all(isinstance(s, int) for s in sequences):
            self.sequences = []
            for s in sequences:
                self.sequences.append("sequence_" + str(s).zfill(3))
        else:
            self.sequences = sequences
        self.sequences = natsorted(self.sequences)

        for sequence in self.sequences:
            sequence_images_dir = Path(self.root) / "raw_data" / sequence
            sequence_semantic_ood_dir = Path(self.root) / "semantic_ood" / sequence
            sequence_instance_ood_dir = Path(self.root) / "instance_ood" / sequence
            sequence_semantic_dir = Path(self.root) / "semantic" / sequence

            sequence_basenames = []
            for file_path in sequence_semantic_ood_dir.glob("*_semantic_ood.png"):
                sequence_basenames.append(
                    str(Path(sequence) / f"{file_path.stem}").replace(
                        "_semantic_ood", ""
                    )
                )
            sequence_basenames = natsorted(sequence_basenames)
            for basename in sequence_basenames:
                self.basenames.append(basename)
                self.images.append(
                    str(sequence_images_dir / f"{Path(basename).stem}_raw_data.png")
                )
                self.targets_semantic_ood.append(
                    str(
                        sequence_semantic_ood_dir
                        / f"{Path(basename).stem}_semantic_ood.png"
                    )
                )
                
                self.targets_instance_ood.append(
                    str(
                        sequence_instance_ood_dir
                        / f"{Path(basename).stem}_instance_ood.png"
                    )
                )
                
                self.targets_semantic.append(
                    str(sequence_semantic_dir / f"{Path(basename).stem}_semantic.png")
                )
                
            for file_path in sequence_images_dir.glob("*.png"):
                self.all_images.append(str(file_path))
                self.all_basenames.append(
                    str(Path(sequence) / file_path.stem.replace("_raw_data", ""))
                )

        self.all_images = natsorted(self.all_images)
        self.all_basenames = natsorted(self.all_basenames)
        self.targets_basenames = [
            self.all_basenames[i] in self.basenames
            for i in range(len(self.all_basenames))
        ]
        
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Draw one input image of dataset
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item, otherwise target is the OOD segmentation by default.
        """
        image = read_image(self.images[index])

        targets: Any = []
        for target_type in self.target_type:
            if target_type == "semantic_ood":
                target = read_image(self.targets_semantic_ood[index],True)
            
            elif target_type == "instance_ood":
                target = read_image(self.targets_instance_ood[index],True)
            
            elif target_type == "semantic":
                target = read_image(self.targets_semantic[index],True)
            
            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        transform = A.Compose([ToTensorV2()])
        aug = transform(image=image, mask=target)
        image = aug['image']
        target = aug['mask']
        target[target == 254] = 1

        return image, target, self.images[index]

    def __len__(self):
        """Denote the total number of samples"""
        return len(self.images)

    def __repr__(self):
        """Print some information about the dataset."""
        fmt_str = "DATASET: Wuppertal Obstacle Scenes\n---- Dir %s\n" % self.root
        fmt_str += "---- Num found images: %d\n" % len(self.images)
        return fmt_str.strip()
