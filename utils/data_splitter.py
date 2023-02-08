import json
import os
import random

from pathlib import Path

def split():
    source_path = Path("/cluster/51/emre/project/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021")
    json_destination_path = Path("/cluster/51/emre/project/3D-MRI-Brain-Tumor-Segmentation-with-Double-U-Net/splits")

    assert os.path.exists(source_path), "source dir does not exist"
    assert os.path.exists(json_destination_path), "destination dir does not exist. Create it first"

    patient_directories = os.listdir(source_path)
    num_patients = len(patient_directories)

    # ensuring operating system invariant ordering
    patient_directories.sort()
    random.seed(42)

    test_size = 0.2
    test = random.sample(patient_directories, k=int(test_size*num_patients))

    # sanity check - no overlap
    train = set(patient_directories).difference(set(test))
    assert len(train.intersection(set(test))) == 0
    train = list(train)
    print(f"length of all labeled data available to us: {num_patients}")
    print(f"length of test set: {len(test)}")
    print(f"length of train set: {len(train)}")

    data = {"test": test, "train": train}

    # 5 splits for train-validation - hyperparameter search
    # in-place shuffling
    random.shuffle(train)

    val_size = 0.2
    num_train = len(train)

    abs_val_size = int(val_size*num_train)
    
    for k in range(5):
        val = train[abs_val_size*k:] if k==4 else train[abs_val_size*k:abs_val_size*(k+1)]
        data[f"split_{k}"] = {"val": val,
                              "train": list(set(train).difference(set(val)))}

    # sanity check
    for k in range(5):
        print(f"fold: {k}")
        print(len(data[f"split_{k}"]["val"]))
        print(len(data[f"split_{k}"]["train"]))
        assert len(set(data[f"split_{k}"]["val"]).intersection(set(data[f"split_{k}"]["train"]))) == 0
        assert len(data[f"split_{k}"]["val"]) + len(data[f"split_{k}"]["train"]) == num_train

    
    with open(json_destination_path / 'datasplit.json', 'w') as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    split()