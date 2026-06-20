import os
import shutil
import pandas as pd

ph_fp = "../misc/Phenotypic_V1_0b.csv"
files_name = ["lh.aparc.stats", "rh.aparc.stats", "aseg.stats"]
data_dest = "../misc/subjects/"
bucket_fp = "/mnt/abide/data/Projects/ABIDE_Initiative/Outputs/freesurfer/5.1/"

def phenotypic_data(filepath):
    df = pd.read_csv(filepath)
    ids = df['FILE_ID']
    df = df[['DX_GROUP', 'AGE_AT_SCAN', 'SEX']]
    return ids, df

def get_data(ids):
    for file_id in ids:
        if file_id == 'no_filename':  
            continue
        
        dest_dir = os.path.join(data_dest, file_id, "stats")
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"Copying {file_id}...")
        for f in files_name:
            src  = os.path.join(bucket_fp, file_id, "stats", f)  # slash manquant
            dst  = os.path.join(dest_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"  Missing: {f}")

def get_all_data():
    ids, pheno = phenotypic_data(ph_fp)
    get_data(ids)
