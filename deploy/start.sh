#!/bin/bash
# Bash script to run the AI Centering Server. Should be called from a systemd Unit File


# -------------- Environment Parameters (MODIFY)
app_cmd=/apps/ai-centering/bin/app.epics  # location where application is installed
epics_env=/apps/profile.d/epics.sh        # EPICS environment setup script

# -------------- Device Parameters (MODIFY)
device="AIC1000-001"                                  # Device Name
appdir="/ioc/instances/${device}"                     # IOC Directory
video="https://video-server.example.com/video.mjpeg"  # Video Stream URI
model="/yolo/models/model-v1.pt"       # Path to YOLO model (ONNX or PyTorch) files

# Prepare environment and  Source function library.
. ${epics_env}
cd $appdir
exec $app_cmd --device $device --video $video --model $model