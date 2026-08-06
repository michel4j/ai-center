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

Running
=======

To run the server directly, use the following command:

```
app.server --device "AIC001" --model "/path/to/model.pt" --video "video_uri"
```

### Command-line Arguments

*   `--device`: The EPICS name of the device (e.g., `AIC001` or `AIC002`). **(Required)**
*   `--model`: Path to the YOLO model file. **(Required)**
*   `--video`: The URI for the video stream. (e.g., `redis://hostname/0030180F06E5`) **(Required)**
*   `--confidence`: Object detection confidence threshold. (Optional, default: 0.1)
*   `-v`: Enable verbose logging. (Optional)

Currently, only `file://`, `redis://` and `http[s]://` video schemes are supported