# Laser Eye : Gaze Estimation via Deep Neural Networks

![BootJump](./asset/logo.webp)

## Easy Start

* [Install Gstreamer](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c)
* Prepare an usb camera
* `chmod 755 ./run.sh`
* `./run.sh`

## Tips
* Change `FRAME_SHAPE` in `demo.py` and `draw.py` to edit the processing image size
* Edit`MxnetDetectionModel`'s `scale` parameter to make a suitable input size of face detector
* More details at [Wiki](https://github.com/1996scarlet/Laser-Eye/wiki)

## A Few Results
![BootJump](./asset/1.gif)
![BootJump](./asset/3.gif)
![BootJump](./asset/4.gif)
![BootJump](./asset/5.gif)
<!-- ![BootJump](./asset/2.gif) -->

## Face Detection
* [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
* [faster-mobile-retinaface (MobileNet Backbone)](https://github.com/1996scarlet/faster-mobile-retinaface)

## Facial Landmarks Detection
* MobileNet-v2 version (1.4MB, using by default)
* [Hourglass2(d=3)-CAB version (37MB)](https://github.com/deepinx/deep-face-alignment)

## Head Pose Estimation
* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

## Iris Segmentation
* [U-Net version (71KB, TVCG 2019)](https://ieeexplore.ieee.org/document/8818661)

## Citation

```
@article{wang2019realtime,
  title={Realtime and Accurate 3D Eye Gaze Capture with DCNN-based Iris and Pupil Segmentation},
  author={Wang, Zhiyong and Chai, Jinxiang and Xia, Shihong},
  journal={IEEE transactions on visualization and computer graphics},
  year={2019},
  publisher={IEEE}
}

@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{Jing2017Stacked,
  title={Stacked Hourglass Network for Robust Facial Landmark Localisation},
  author={Jing, Yang and Liu, Qingshan and Zhang, Kaihua and Jing, Yang and Liu, Qingshan and Zhang, Kaihua and Jing, Yang and Liu, Qingshan and Zhang, Kaihua},
  booktitle={IEEE Conference on Computer Vision & Pattern Recognition Workshops},
  year={2017},
}
```
