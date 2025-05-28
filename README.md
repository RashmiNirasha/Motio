# ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild

This is the official Github repository for our paper **[ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild][arxiv-link]**, published at ECCV Workshops, 2024.

**Authors:** [Arya Farkhondeh](https://scholar.google.com/citations?hl=en&user=ZgMfa7gAAAAJ&view_op=list_works&sortby=pubdate)\*,
[Samy Tafasca](https://scholar.google.com/citations?user=N6SqiAIAAAAJ&hl=en)\*,
[Jean-Marc Odobez](https://scholar.google.com/citations?user=iHYNhAkAAAAJ&hl=en)

\*: Equal contribution.

**ChildPlay-Hand** is a novel video dataset for modeling Hand-Object Interaction (HOI) in the wild that includes person and object bounding boxes, as well as manipulation actions. ChildPlay-Hand is unique in: **(1)** providing per-hand annotations; **(2)** featuring videos in uncontrolled settings with natural interactions; **(3)** including gaze labels from the ChildPlay-Gaze dataset for joint modeling of manipulations and gaze. We introduce two tasks: object-in-hand (OiH) and manipulation stages (ManiS), and benchmark various spatio-temporal and segmentation networks on these tasks.

[`Paper`](https://arxiv.org/abs/2409.09319) | [`BibTeX`](./README.md#citation)

![image info](./assets/childplay_hand_cover.png)

## Setup

### 1. Create the Conda environment

See `SETUP.md`

### 2. Organize the data artifacts

See `data/README.md` for how to download and organize the data and annotations.

## Demo

We provide a demo to test the models on your input video. The demo generates a video with predicted hand actions and, optionally, the extracted 2D skeletons overlaid for all individuals. To run the demo:

```shell
python demo.py video_path=path/to/my_video.mp4 output_path="my_video"
```
To customize the settings, edit the configuration file at `configs/demo.yaml`.

## Training
This project uses PyTorch Lightning for training and Hydra for managing configurations. Each model-task combination has a dedicated config file located at `./configs/{model}_{task}.yaml`. You also need to specify the input type explicitly. The available options are:

- `model`: {poseconv3d, rgbposeconv3d, hiera, sstcn, mstcn}
- `task`: {object, manipulation}
- `input_type`: {body, hand}

Here is how you can run a training job using the Hiera model with hand input (Hiera-Hand) for the manipulation (ManiS) task:
```shell
python train.py --config-name hiera_manipulation.yaml configs.input_type=hand
```

## Testing
To run the test and reproduce the results reported in the paper, execute:

```shell
bash test.sh
```
This will generate the evaluation outputs as reported in the paper. You can specify different models, tasks, and input types by editing the `test.sh` script.

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

## Acknowledgement
Parts of the code were adapted from the repositories [PYSKL](https://github.com/kennymckormick/pyskl) and [LART](https://github.com/brjathu/LART). We are thankful to the authors for their contributions.

[arxiv-link]: https://arxiv.org/abs/2409.09319