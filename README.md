aicenter
========

A python based Soft IOC Server for Sample Alignment using a YOLO model.

Installation
============

```
python -m venv my-venv
source my-venv/bin/activate
pip install ai-center
```

To enabled downloading of the latest model file from HuggingFace, install with the `[model]` extra as follows:

```
pip install ai-center[model]
```

Running
=======

To run the EPICS IOC directly, use the following command:

```
app.epics --device "AIC001" --model "/path/to/model.pt" --video "video_uri"
```

### Command-line Arguments

*   `--device`: The EPICS root name of the device (e.g., `AIC001` or `AIC002`). **(Required)**
*   `--model`: Path to the YOLO model file. If omitted, the latest model will be downloaded from HuggingFace. The model is about 50 MB in size.
*   `--video`: The URI for the video stream. (e.g., `redis://hostname/0030180F06E5:JPG`) **(Required)**
*   `--confidence`: Object detection confidence threshold. (Optional, default: 0.1)
*   `-v`: Enable verbose logging. (Optional)

Currently, only `file://`, `redis://` and `http[s]://` video schemes are supported


If you installed the `[view]` extra, you can also run the stand-alone inference viewer which displays the inference in a
window as follows:

```
app.view --model "/path/to/model.pt" --video "video_uri"
```

### Command-line Arguments

*   `--model`: Path to the YOLO model file. If omitted, the latest model will be downloaded from HuggingFace. The model is about 50 MB in size.
*   `--video`: The URI for the video stream. (e.g., `redis://hostname/0030180F06E5:JPG`) **(Required)**
*   `--confidence`: Object detection confidence threshold. (Optional, default: 0.1)
*   `-v`: Enable verbose logging. (Optional)