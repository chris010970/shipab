@echo off
set GCS_BUCKET=gs://eo-ald-update
set REPO=shipab

gsutil cp %GCS_BUCKET%/%REPO%/data/subset-1.zip .
