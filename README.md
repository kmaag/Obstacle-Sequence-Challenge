## Datasets

- [Street Obstacle Sequences (SOS)](https://zenodo.org/records/7144906)
- [Wuppertal Obstacle Sequences (WOS)](https://zenodo.org/records/12188586)

## Evaluation 

### Preparation:

For each image of both datasets (e.g. `street_obstacle_sequences/raw_data/sequence_001/000000_raw_data.jpg`), the following data is required:
- OOD scores (height, width) as numpy array, saved for example as `street_obstacle_sequences/ood_score/sequence_001/000000.npy` 
- Object tracking IDs (height, width) as numpy array, saved for example as `street_obstacle_sequences/ood_prediction_tracked/sequence_001/000000.npy` 

### Metrics:

To compute the evaluation metrics run

```python
python main.py --dataset_path ./datasets/street_obstacle_sequences --dataset sos
```
for the SOS dataset and 
```python
python main.py --dataset_path ./datasets/wos --dataset wos
```
for the WOS dataset.

## Citation

If you use this repository, please consider citing our [paper](https://openaccess.thecvf.com/content/ACCV2022/html/Maag_Two_Video_Data_Sets_for_Tracking_and_Retrieval_of_Out_ACCV_2022_paper.html):

    @InProceedings{Maag_2022_ACCV,
        author    = {Maag, Kira and Chan, Robin and Uhlemeyer, Svenja and Kowol, Kamil and Gottschalk, Hanno},
        title     = {Two Video Data Sets for Tracking and Retrieval of Out of Distribution Objects},
        booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
        month     = {December},
        year      = {2022},
        pages     = {3776-3794}
    }


