#!/bin/bash
sudo umount -l /mnt/abide
sudo s3fs fcp-indi /mnt/abide \
  -o public_bucket=1 \
  -o endpoint=us-east-1 \
  -o url=https://s3.amazonaws.com \
  -o allow_other
ln -sf /mnt/abide/abide/data/Projects/ABIDE_Initiative abide
