#!/bin/bash
# Bash script to run the AI Centering Server. Should be called from a systemd Unit File


# -------------- Environment Parameters (MODIFY)
app_cmd=/apps/ai-centering/bin/app.server  # location where application is installed
epics_env=/apps/profile.d/epics.sh         # EPICS environment setup script

# -------------- Device Parameters (MODIFY)
device="AIC1000-001"                                # Device Name
appdir="/ioc/instances/${device}"                   # IOC Directory
server="video-server.example.com"                   # Redis Video Server hostname or IP address
camera="0030180F06E5"                               # Redis Video Stream ID
model="/yolo/models/scrn-20250417-crystals-v8m"     # Path to YOLO model (ONNX or Darknet) files

# Prepare environment and  Source function library.
. ${epics_env}
cd $appdir
exec $app_cmd --device $device --server $server --camera $camera --model $model