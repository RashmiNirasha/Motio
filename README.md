# Motio: Analysis of Child Hand-Object Interaction Using the ChildPlay-Hand Dataset

This project builds on the ChildPlay-Hand dataset released by the original authors of the dataset and paper. The data itself remains the property of that research team, and this repository is a focused analysis and implementation built on top of their annotations and recordings.

## Dataset source and attribution

The underlying dataset is the ChildPlay-Hand dataset introduced by:

- Arya Farkhondeh
- Samy Tafasca
- Jean-Marc Odobez

This work was originally presented in the paper:

- ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild
- ECCV Workshops, 2024

The dataset provides child hand and object interaction recordings with annotations that make it suitable for studying hand-object interaction in naturalistic settings.

## Project goal

This repository analyzes the ChildPlay-Hand data for the task of hand-object interaction detection. The objective is to turn frame-level hand landmarks and temporal motion patterns into a signal that can distinguish active object interaction from non-interaction in a lightweight, interpretable way.

## My analysis

The analysis in this project focuses on the following pipeline:

1. Feature extraction from hand landmarks using normalized wrist-relative coordinates
2. Construction of temporal sequences from consecutive frames
3. Standardization of landmark features using training statistics
4. Temporal modeling of hand motion and position over a fixed window
5. Threshold-based decision logic for interaction detection
6. Real-time inference for live monitoring via MediaPipe hand tracking

### Key modeling choices

- The landmark structure is normalized relative to the wrist and hand bounding box, preserving motion patterns while reducing dependence on absolute frame position.
- Temporal context is captured using a 20-frame sequence, which is suited for short interactions and repeated motion patterns.
- The model predicts interaction probability and applies a calibrated threshold to decide whether the hand is likely interacting with an object.
- Live monitoring uses MediaPipe to estimate landmarks in real time, then converts them into the same feature form used during model training.

### Analysis outputs

The project includes:

- dataset processing utilities for selecting and examining candidate grasp events
- model training scripts for temporal interaction classification
- threshold tuning and confusion-matrix evaluation
- a FastAPI backend for serving predictions
- a browser-based live monitoring interface using the webcam

### Practical interpretation

The central finding is that child hand-object interaction can be detected effectively using short temporal sequences of hand landmarks rather than requiring full scene-level object detection. This makes the system more lightweight and suitable for real-time monitoring use cases where the goal is to identify interaction events from hand motion dynamics.

## Repository contents

This project contains:

- data preparation and filtering scripts
- dataset inspection and annotation utilities
- temporal model training and threshold tuning code
- an MVP web app for live interaction detection
- supporting assets and evaluation outputs

## Setup

For local setup and environment configuration, see:

- `SETUP.md`
- `data/README.md`

## Relevant references

- ChildPlay-Hand paper: https://arxiv.org/abs/2409.09319
- Related ChildPlay benchmark work: https://openaccess.thecvf.com/content/ICCV2023/html/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Children%27s_Gaze_Behaviour_ICCV_2023_paper.html

## Acknowledgement

This work depends on the original ChildPlay-Hand dataset and research contributions from the dataset authors. The analysis and implementation in this repository are my own project work built on top of their data and benchmark foundation.

