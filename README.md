
aicenter
========

A python based Soft IOC Server for Sample Alignment using a YOLO model.

Usage
=====

```
python -m venv my-venv
source my-venv/bin/activate
pip install ai-center
```

1. Create a directory for the IOC instance.
2. Copy the start.sh file from the `deploy` directory into this directory.
3. Copy the ai-centering.service unit file from the `deploy` into your /etc/systemd/system directory.
4. Edit the files from (2) and (3) above to reflect your environment and to set all the required instance parameters
4. Enable the unit file using your system commands. For example, `systemctl enable ai-centering`.
5. Start the init file using your system commands. For example `systemctl start ai-centering`.

You can manage the instance daemon through procServ, by telneting to the configured port. 

Installation
============

```
pip install .[ioc]
```

OpenCV
======

For best performance, a version of `python-opencv` compiled with support for CUDA and cuDNN along
with a compatible GPU should be used.

Testing
=======

The `test/inference.py` file can be used to test the inference / model performance without running
a full IOC application.  To obtain a copy of the trained model weights, contact the authors.

Install `aicenter` without `[ioc]` dependencies:

```
pip install .[test]
```

Segment Anything
================

To enable segmentation tracking with SAM2, install with `[sam]` extra.

Model weights must be downloaded from the [sam2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#model-description)
page. Currently checkpoint files for SAM 2 (July 2024) are supported.

By default, the `aicenter.sam` module looks for weights in `<my-aicenter-venv>/sam_weights/sam2_hiera_large.pt`

### Acknowledgment
SAM support uses [muggled_sam](https://github.com/heyoeyo/muggled_sam) which itself is an
implementation of Segment Anything 2:

[facebookresearch/sam2](https://github.com/facebookresearch/sam2)
```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
