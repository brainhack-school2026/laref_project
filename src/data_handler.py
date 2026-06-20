import joblib
import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from classifier import TEST_SIZE, RANDOM_STATE
 

# hemispheric labels
aparc_regions = [
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal',
    'cuneus', 'entorhinal', 'fusiform', 'inferiorparietal',
    'inferiortemporal', 'isthmuscingulate', 'lateraloccipital',
    'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
    'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis',
    'pericalcarine', 'postcentral', 'posteriorcingulate',
    'precentral', 'precuneus', 'rostralanteriorcingulate',
    'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal',
    'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula'
]

# volumetric volumetric labels
aseg_regions = [
    'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
    'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
    'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
    'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 'Left-choroid-plexus',
    'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter',
    'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate',
    'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
    'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus',
    '5th-Ventricle', 'WM-hypointensities', 'Left-WM-hypointensities',
    'Right-WM-hypointensities', 'non-WM-hypointensities', 'Left-non-WM-hypointensities',
    'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior',
    'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior'
]




data_path = "../misc/subjects/"


# Cortical thickness (left hemisphere vs right hemisphere)
lh_fp = "/stats/lh.aparc.stats" # data_path + test_subject + "lh.aparc.stats"
rh_fp = "/stats/rh.aparc.stats" # data_path + test_subject + "rh.aparc.stats"

# Volumetric statistics for the brain
aseg_fp = "/stats/aseg.stats" # data_path + test_subject + "aseg.stats"

# phenotypic data
ph_fp = "../misc/Phenotypic_V1_0b.csv"


CLEAN_DATA_PATH  = "../misc/abide_features.csv"
SCALER_OUT = "../misc/scaler.pkl"
PREPARED_DATA   = "../misc/abide_prepared.npz"
FEATS_OUT  = "../misc/feature_cols.pkl"
MODEL_OUT  = "../misc/brain_model.pt"



def hemispheric_data(filepath):
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        comment='#',
        header=None,
        names=[
            'StructName', 'NumVert', 'SurfArea', 'GrayVol',
            'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv',
            'FoldInd', 'CurvInd'
        ]
    )
    df.drop(columns=['StructName'], inplace=True)
    return df

def volumetric_data(filepath):
    skip = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith('#') or line.strip() == '':
                skip.append(i)
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        skiprows=skip,
        header=None,
        names=[
            'Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName',
            'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange'
        ]
    )
    df.drop(columns=['Index', 'SegId', 'StructName', 'NVoxels'], inplace=True)
    return df

def phenotypic_data():
    df = pd.read_csv(ph_fp)
    ids = df['FILE_ID']
    df = df[['DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'SITE_ID']]
    return ids, df

def load_subject(subject_name, pheno_row):
    try:
        lh_data   = hemispheric_data(data_path + subject_name + lh_fp)
        rh_data   = hemispheric_data(data_path + subject_name + rh_fp)
        aseg_data = volumetric_data(data_path + subject_name + aseg_fp)
    except FileNotFoundError:
        print(f"Skipping {subject_name} — fichier manquant")
        return None

    row = {}
    row['DX_GROUP']    = pheno_row['DX_GROUP']
    row['AGE_AT_SCAN'] = pheno_row['AGE_AT_SCAN']
    row['SEX']         = pheno_row['SEX']
    row['SITE_ID']         = pheno_row['SITE_ID']


    for col in lh_data.columns:
        for region, val in zip(aparc_regions, lh_data[col]):
            row[f"lh_{region}_{col}"] = val

    for col in rh_data.columns:
        for region, val in zip(aparc_regions, rh_data[col]):
            row[f"rh_{region}_{col}"] = val

    for col in aseg_data.columns:
        for region, val in zip(aseg_regions, aseg_data[col]):
            row[f"aseg_{region}_{col}"] = val

    return row

def load_all_subjects():
    ids, pheno = phenotypic_data()
    pheno['FILE_ID'] = ids.values
    rows = []

    for _, pheno_row in pheno.iterrows():
        subject_name = pheno_row['FILE_ID']
        if subject_name == 'no_filename':
            continue
        row = load_subject(subject_name, pheno_row)
        if row is not None:
            rows.append(row)

    subject_db = pd.DataFrame(rows).fillna(0)
    print(f"Loaded {len(subject_db)} subjects, {len(subject_db.columns)} features")
    return subject_db




# we will use neuro combat to hamonize our data across the differente sites
def harmonize(subject_db):

    feature_cols = [c for c in subject_db.columns 
                    if c not in ['DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'SITE_ID']]

    data = subject_db[feature_cols].values.T

    # we will keep these variables unchanged across the sitee
    covars = pd.DataFrame({
        'SITE_ID':      subject_db['SITE_ID'].values,
        'AGE_AT_SCAN':  subject_db['AGE_AT_SCAN'].values,
        'SEX':          subject_db['SEX'].values,
        'DX_GROUP':     subject_db['DX_GROUP'].values,
    })
    

    data_harmonized = neuroCombat(
        dat=data,
        covars=covars,
        batch_col='SITE_ID',                        
        categorical_cols=['SEX', 'DX_GROUP'],       
        continuous_cols=['AGE_AT_SCAN']             
    )['data']
    
    # Reconstruire le DataFrame
    df_harmonized = pd.DataFrame(
        data_harmonized.T,
        columns=feature_cols
    )
    df_harmonized['DX_GROUP']   = subject_db['DX_GROUP'].values
    df_harmonized['AGE_AT_SCAN'] = subject_db['AGE_AT_SCAN'].values
    df_harmonized['SEX']        = subject_db['SEX'].values - 1


    df_harmonized.to_csv(CLEAN_DATA_PATH, index=False)

    return df_harmonized





# after this function the data is ready for training
def prepare_data(data):

    feature_cols = [c for c in data.columns if c != 'DX_GROUP']
    X = data[feature_cols].values.astype(np.float32)
    y = (data['DX_GROUP'].values == 1).astype(np.float32)  # 1=ASD, 0=TD
 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
 
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test  = y[test_idx]
 
    np.savez(PREPARED_DATA, X_train=X_train, X_test=X_test,
                             y_train=y_train, y_test=y_test)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(feature_cols, FEATS_OUT)
 
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"Saved: {PREPARED_DATA}, {SCALER_OUT}, {FEATS_OUT}")
 

