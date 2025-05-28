# ChildPlay-Hand Dataset

Instructions for downloading and organizing the required data: `videos`, `annotations`, `checkpoints`, `feats`, and `pose` needed for training and evaluation.

---

## Videos

ChildPlay-Hand builds on videos from the ChildPlay-Gaze dataset. Download the videos from the [ChildPlay-Gaze GitHub](https://github.com/idiap/geomgaze). Follow their instructions to extract frames from the videos. Place the dataset under the `data/` directory so you have:

```markdown
data/ChildPlay-gaze/
└── images/  # extracted video frames
└── ... 
```

## Annotation

Download the annotations from [Zenodo](https://zenodo.org/records/14958923) (5.1GB) and unzip them under the `data/` directory.  This will create the following structure:

```markdown
data/childplay_hand/
├── data
│   ├── annotations
│   │   ├── train
│   │   │   ├── 1Aea8BH-PCs_1256-1506.csv
│   │   │   ├── 1Aea8BH-PCs_3248-3602.csv
│   │   │   └── ...
│   │   ├── val
│   │   │   └── ...
│   │   └── test
│   │       └── ...
│   ├── feats
│   │   ├── childplay_hiera_hand_{task}_{split}.pkl
│   │   └── ...
│   ├── pose
│   │   ├── cp_pose_{split}.pkl
│   │   └── ...
├── checkpoints
│   ├── hiera_manipulation_body.ckpt
│   ├── hiera_manipulation_hand.ckpt
│   └── ...
```

### annotations 

Contains `.csv` annotation files, each corresponding to a specific video clip. Files are organized by data split (train, val, test). Each CSV file has following columns:

- **`clip`**: Name of the video clip.
- **`frame`**: Frame ID.
- **`pid`**: Person ID.
- **`p_xmin, p_ymin, p_xmax, p_ymax`**: Ground truth (GT) coordinates of the body bounding box.
- **`oid`**: Object ID.
- **`o_xmin, o_ymin, o_xmax, o_ymax`**: GT coordinates of the object bounding box being interacted with.
- **`hand`**: Hand side, i.e., left or right.
- **`p_visible`**: Person visibility (0 or 1).
- **`h_visible`**: Pseudo hand visibility score between 0 and 1.
- **`interaction`**: GT hand labels in string format (10 classes in ChildPlay-Hand).
- **`manipulation_task`**: GT hand labels for the manipulation (ManiS) task (integer).
- **`object_task`**: GT hand labels for the object-in-hand (OiH) task (integer).
- **`full_task`**: GT hand labels for the full task (integer).

### feats

Contains extracted features for temporal segmentation tasks, saved as `.pkl` files. The files are named as `childplay_hiera_hand_{task}_{split}.pkl`, where `{task}` specifies the task type (e.g., object, manipulation) and `{split}` indicates the dataset split (train, val, test). 

```python
import joblib

feat = joblib.load("data/feats/childplay_hiera_hand_manipulation_val.pkl")

>>> feat.keys()
dict_keys(['bI1GohGXSt0_4992-5101', 'Kaf6jiHMmYc_6757-6910', 'bI1GohGXSt0_40031-40552', ...]) # Video Names

>>> feat['bI1GohGXSt0_4992-5101'].keys()
dict_keys([1, 2]) # Person IDs

>>> feat['bI1GohGXSt0_4992-5101'][1].keys()
dict_keys(['left', 'right'])

>>> feat['bI1GohGXSt0_4992-5101'][1]['left'].keys()
dict_keys(['feats'])
```
- **`feats`**: Extracted features. Shape: (n_frames, dim), where `n_frames` is number of frames of the video and `dim` is the feature vector dim. 

### pose

Contains pose data for each split, saved as `.pkl` files. Each file is named `cp_pose_{split}.pkl`, where `{split}` represents the data split (train, val, test). These files include detailed information on hand or body poses relevant to each task. If needed, you can generate this data using the Python script `src/utils/pose_extraction.py`.

```python
import joblib

pose = joblib.load("data/pose/cp_pose_val.pkl")

>>> pose.keys()
dict_keys(['bI1GohGXSt0_4992-5101', 'Kaf6jiHMmYc_6757-6910', 'bI1GohGXSt0_40031-40552', ...]) # Video Names

>>> pose['bI1GohGXSt0_4992-5101'].keys()
dict_keys([1, 2]) # Person IDs

>>> pose['bI1GohGXSt0_4992-5101'][1].keys()
dict_keys(['img_shape', 'body_keypoint', 'body_keypoint_score', 'body_bbox', 'left_hand_keypoint', 'right_hand_keypoint', 'left_hand_keypoint_score', 'right_hand_keypoint_score', 'left_hand_bbox', 'right_hand_bbox'])
```

- **`img_shape`**: Shape of the input frames in ChildPlay. 
- **`body_keypoint`**: Extracted 17 body keypoints using MMPose (refer to `src/utils/pose_extraction.py`). Shape: (1, n_frames, 17, 2).
- **`body_keypoint_score`**: Confidence scores for the extracted 17 body keypoints. Shape: (1, n_frames, 17).
- **`body_bbox`**: Ground truth (GT) coordinates of the body bounding box. Shape: (1, n_frames, 4).
- **`{hand_side}_hand_keypoint`**: Extracted 21 keypoints of the left hand using MMPose (refer to `src/utils/pose_extraction.py`). Shape: (1, n_frames, 21, 2).
- **`{hand_side}_hand_keypoint_score`**: Confidence scores for the left hand keypoints. Shape: (1, n_frames, 21).
- **`{hand_side}_hand_bbox`**: Pseudo bounding box for the left hand (refer to `src/utils/generate_hand_bbox.py`). Shape: (1, n_frames, 4).

ℹ️ Here, `{hand_side}` is either `left` or `right`.

### checkpoints

We provide all checkpoints. You can download them from [here](https://zenodo.org/records/14958923). Each checkpoint follows the format `{model}_{task}_{input_type}.ckpt` with the following options:

- **`model`**: `{poseconv3d, rgbposeconv3d, hiera}`
- **`task`**: `{object, manipulation}`
- **`input_type`**: `{body, hand}`

## Citation

If you use our code, models or data assets, please consider citing us:

```bibtex
@inproceedings{Farkhondeh_ECCVW_2024,
  author    = {Farkhondeh*, Arya and Tafasca*, Samy and Odobez, Jean-Marc},
  title     = {ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
  year      = {2024},
  note      = {* Equal contribution}
}

@InProceedings{Tafasca_2023_ICCV,
    author    = {Tafasca*, Samy and Gupta*, Anshul and Odobez, Jean-Marc},
    title     = {ChildPlay: A New Benchmark for Understanding Children's Gaze Behaviour},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20935-20946},
    note      = {* Equal contribution}
}
```