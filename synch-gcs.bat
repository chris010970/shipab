@echo off
set GCS_BUCKET=gs://eo-ald-update
set ROOT_PATH=c:\Users\Chris.Williams\Documents\GitHub
set REPO=shipab

REM --------------- plant type ---------------
gsutil cp %ROOT_PATH%\%REPO%\models\mask-rcnn.zip %GCS_BUCKET%/%REPO%/models/mask-rcnn.zip
gsutil cp %ROOT_PATH%\%REPO%\data\subset-1.zip %GCS_BUCKET%/%REPO%/data/subset-1.zip