
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

Segment Anything
================

To enable segmentation tracking with SAM2, install with `[sam]` extra.

Testing
=======

The `test/inference.py` file can be used to test the inference / model performance without running
a full IOC application.  To obtain a copy of the trained model weights, contact the authors.

Install `aicenter` without `[ioc]` dependencies:

```
pip install .[test]
```
