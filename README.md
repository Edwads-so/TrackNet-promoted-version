# TrackNet-promoted-version: Lightweight, Motion-Aware TrackNet for High-Speed Small Object Tracking

An improved version of [TrackNetV3](https://github.com/qaz812345/TrackNetV3) for tracking small, fast-moving objects (e.g. a badminton shuttlecock) in video. The original TrackNetV3 pipeline (tracking + trajectory rectification) is kept, and a new tracking network, **RefinedTrackNet**, is added (`refined_model.py`).

## What's new compared with TrackNetV3

| Change | Where | Motivation |
|---|---|---|
| **Depthwise-separable convolutions** replace the standard conv blocks in the encoder/decoder | `DSConv2DBlock`, `LightweightDoubleConv` | Cut parameters and FLOPs to raise FPS |
| **Adaptive Motion Module**: explicit frame differences between the 3 stacked input frames, re-weighted by a learned gate | `AdaptiveMotionModule` | Suppress static background noise and reduce false positives |
| **Improved MaSA bottleneck**: spatial attention that keeps resolution instead of mean-pooling, plus a local enhancement mask | `ImprovedMaSA` (based on `MaSA_model.py`) | Keep small-object features from being diluted |
| Motion features fused into the bottleneck, U-Net skip connections kept | `RefinedTrackNet` | Combine appearance and motion cues |

Train it with `--model_name refined` (`--model_name TrackNet` still trains the original network). The rectification module (InpaintNet) is unchanged.

## Installation
* Develop Environment
    ```
    Python 3.10
    torch 2.3.1 (see requirements.txt)
    ```
* Clone this repository.
    ```
    git clone https://github.com/Edwads-so/TrackNet-promoted-version.git
    cd TrackNet-promoted-version
    ```

* Install the requirements.
    ```
    pip install -r requirements.txt
    ```

## Inference
* Download the original TrackNetV3 [checkpoints](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing)
* Unzip the file and place the parameter files to ```ckpts```
    ```
    unzip TrackNetV3_ckpts.zip
    ```
* Predict the label csv from the video
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction
    ```
* Predict the label csv from the video, and output a video with predicted trajectory
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --output_video
    ```
* For large video
    * Enable the ```--large_video``` flag to use an IterableDataset instead of the normal Dataset, which prevents memory errors. Note that this will decrease the inference speed.
    * Use ```--max_sample_num``` to set the number of samples for background estimation.
    * Use ```--video_range``` to specify the start and end seconds of the video for background estimation.
    ```
    python predict.py --video_file test.mp4 --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir prediction --large_video --video_range 324,330
    ```

## Training
### 1. Prepare Dataset
* Download [Shuttlecock Trajectory Dataset](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw)
* Adjust file structure:
    1. Merge the `Professional` and `Amateur` match directories into a single `train` directory.
    2. Rename the `Amateur` match directories to start from `match24` through `match26`.
    3. Rename the `Test` directory to `test`.
* Dataset file structure:
```
  data
    ├─ train
    |   ├── match1/
    |   │   ├── csv/
    |   │   │   ├── 1_01_00_ball.csv
    |   │   │   ├── 1_02_00_ball.csv
    |   │   │   ├── …
    |   │   │   └── *_**_**_ball.csv
    |   │   ├── frame/
    |   │   │   ├── 1_01_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── 1_02_00/
    |   │   │   │   ├── 0.png
    |   │   │   │   ├── 1.png
    |   │   │   │   ├── …
    |   │   │   │   └── *.png
    |   │   │   ├── …
    |   │   │   └── *_**_**/
    |   │   │
    |   │   └── video/
    |   │       ├── 1_01_00.mp4
    |   │       ├── 1_02_00.mp4
    |   │       ├── …
    |   │       └── *_**_**.mp4
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    ├─ val
    |   ├── match1/
    |   ├── match2/
    |   │ ⋮
    |   └── match26/
    └─ test
        ├── match1/
        ├── match2/
        └── match3/
```
* Attributes in each csv files: `Frame, Visibility, X, Y`
* Data preprocessing
    ```
    python preprocess.py
    ```
* The `frame` directories and the `val` directory will be generated after preprocessing.
* Check the estimated background images in `<data_dir>/median`
    * If available, the dataset will use the median image of the match; otherwise, it will use the median image of the rally.
    * For example, you can exclude `train/match16/median.npz` due to camera angle discrepancies; therefore, the dataset will resort to the median image of the rally within match 16.
* Set the data root directory to `data_dir` in `dataset.py`.
    * `dataset.py` will generate the image mapping for each sample and cache the result in `.npy` files.
    * If you modify any related functions in `dataset.py`, please ensure you delete these cached files.
### 2. Train Tracking Module
* Train the improved tracking module (RefinedTrackNet) from scratch
    ```
    python train.py --model_name refined --seq_len 3 --epochs 30 --batch_size 10 --save_dir exp --verbose
    ```

* Train the original TrackNetV3 tracking module from scratch
    ```
    python train.py --model_name TrackNet --seq_len 8 --epochs 30 --batch_size 10 --bg_mode concat --alpha 0.5 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name TrackNet --epochs 30 --save_dir exp --resume_training --verbose
    ```

### 3. Generate Predicted Trajectories and Inpainting Masks
* Generate predicted trajectories and inpainting masks for training rectification module
    * Note that the coordinate range corresponds to the input spatial dimensions, not the size of the original image.
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --batch_size 16
    ```

### 4. Train Rectification Module
* Train the rectification module from scratch.
    ```
    python train.py --model_name InpaintNet --seq_len 16 --epochs 300 --batch_size 32 --lr_scheduler StepLR --mask_ratio 0.3 --save_dir exp --verbose
    ```

* Resume training (start from the last epoch to the specified epoch)
    ```
    python train.py --model_name InpaintNet --epochs 30 --save_dir exp --resume_training
    ```

## Evaluation
* Evaluate TrackNetV3 on test set
    ```
    python generate_mask_data.py --tracknet_file ckpts/TrackNet_best.pt --split_list test
    python test.py --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval
    ```

* Evaluate the tracking module on test set
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --save_dir eval
    ```

* Generate video with ground truth label and predicted result
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --video_file data/test/match1/video/1_05_02.mp4 
    ```

## Error Analysis Interface
* Evaluate TrackNetV3 on test set and save the detail results for error analysis
    ```
    python test.py --tracknet_file ckpts/TrackNet_best.pt --inpaintnet_file ckpts/InpaintNet_best.pt --save_dir eval --output_pred
    ```

* Add json path of evaluation results to the file list in `error_analysis.py`
    ```
    30  # Evaluation result file list
    31  if split == 'train':
    32      eval_file_list = [
    33          {'label': label_name, 'value': json_path},
     ⋮                              ⋮
            ]
        elif split == 'val':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        elif split == 'test':
            eval_file_list = [
                {'label': label_name, 'value': json_path},
                                    ⋮
            ]
        else:
            raise ValueError(f'Invalid split: {split}')                                  
    ```

* Run Dash application
    ```
    python error_analysis.py --split test --host 127.0.0.1
    ```

## Reference
* TrackNetV3 (base of this project): https://github.com/qaz812345/TrackNetV3 [[paper](https://dl.acm.org/doi/10.1145/3595916.3626370)]
* TrackNetV2: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
* Shuttlecock Trajectory Dataset: https://hackmd.io/@TUIK/rJkRW54cU
* Labeling Tool: https://github.com/Chang-Chia-Chi/TrackNet-Badminton-Tracking-tensorflow2?tab=readme-ov-file#label
