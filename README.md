
# ASD Classifier based on Structural Brain Imaging Data

This project trains a classifier to distinguish Autism Spectrum Disorder (ASD) cases from typical controls using structural MRI features from the [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) (Autism Brain Imaging Data Exchange) dataset.

A report is also available : [report.pdf](https://github.com/brainhack-school2026/laref_project/blob/main/report.pdf).


## Project structure

```
.
├── abide -> /mnt/abide/abide/data/Projects/ABIDE_Initiative   # symlink to mounted S3 data
├── misc/                                                       
├── mount_data.sh                                               # mounts the ABIDE S3 bucket
├── report.pdf                                                 
├── requirements.txt
└── src/
    ├── main.py            #  entry point
    ├── data_import.py     # imports raw subject data
    ├── data_handler.py     # loads, harmonizes, and prepares subject data
    ├── classifier.py       # model definition
    ├── train.py            # training loop
    └── analysis.py         # evaluation / results analysis

```

## Setup

Create and activate a virtual environment, then install dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

```

## Getting the data

This repository ships with the data already included under `misc/`, so you can go straight to [Usage](#usage) without fetching anything yourself.

If you'd rather fetch it yourself from the source (for example, to verify provenance or get an updated copy), the data is hosted on a public Amazon S3 bucket and can be mounted locally with [`s3fs`](https://github.com/s3fs-fuse/s3fs-fuse):

1.  Install `s3fs` (e.g. `sudo apt install s3fs` on Debian/Ubuntu).
    
2.  Create the mount point and mount the bucket:
    
    ```sh
    sudo mkdir -p /mnt/abide
    ./mount_data.sh
    
    ```
    
    This mounts the public `fcp-indi` bucket to `/mnt/abide` and creates a local symlink, `abide`, pointing into the ABIDE Initiative project folder.
    
3.  Fetch and prepare the data through the CLI:
    
    ```sh
    cd src
    python main.py fetch
    
    ```
    
    Fetching the full dataset from the bucket can take several hours, which is why a local copy is included in this repo by default.
    

## Usage

All commands are run from the `src/` directory:

```sh
cd src
python main.py [fetch|process|train|analyse]

```

Command

Description

`fetch`

Imports phenotypic and imaging data (from `misc/` or the mounted bucket).

`process`

Loads all subjects, harmonizes the data across sites, and prepares it for training.

`train`

Trains the classifier.

`analyse`

Runs evaluation and analysis on the trained model and produces results.

Typical pipeline:

```sh
python main.py fetch
python main.py process
python main.py train
python main.py analyse

```

## Requirements

See `requirements.txt` for the full list of Python dependencies (PyTorch, pandas, etc.).

