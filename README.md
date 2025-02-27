#  MSWAL : 3D Multi-class Segmentation of Whole Abdominal Lesions Dataset

This repo presents the implementation of the **MSWAL 🩻** 

## Details
With the significantly increasing incidence and prevalence of abdominal diseases, there is a need to embrace greater use of new innovations and the technology for the diagnosis and treatment of patients. Although deep-learning methods have notably been developed to assist radiologists in diagnosing abdominal diseases, existing models have the restricted ability to segment common lesions in the abdomen due to missing annotations for typical abdominal pathologies in their training datasets. To address the limitation, we introduce **MSWAL**, the first 3D **M**ulti-class **S**egmentation of the **W**hole **A**bdominal **L**esions dataset, which broadens the coverage of various common lesion types, such as gallstones, kidney stones, liver tumors, kidney tumors, pancreatic cancer, liver cysts, and kidney cysts. With CT scans collected from 694 patients (191,417 slices) of different genders across various scanning phases, MSWAL demonstrates strong robustness and generalizability. The transfer learning experiment from MSWAL to two public datasets, LiTS and KiTS, effectively demonstrates consistent improvements, with Dice Similarity Coefficient (DSC) increase of 3.00\% for liver tumors and 0.89\% for kidney tumors, demonstrating that the comprehensive annotations and diverse lesion types in MSWAL facilitate effective learning across different domains and data distributions. Furthermore, we propose **Inception nnU-Net**, a novel segmentation framework that effectively integrates an Inception module with the nnU-Net architecture to extract information from different receptive fields, achieving significant enhancement in both voxel-level DSC and region-level F1 compared to the cutting-edge public algorithms on MSWAL.



## Get Started
Inception nnU-Net in this paper is highly dependent on the preprocessing and architecture of nnU-Net. You can find how nnU-Net works here: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

Then we will introduce how to run Inception nnU-Net.

## Step 1: install nnU-Net.
As shown in [nnU-Net installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md), please install pytorch firstly with python higher than 3.9. Then install nnU-Net.
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

## Step 2: preprocess the dataset.
You should create a folder to store the dataset, which has three sub-folders: nnU-Net_raw; nnUNet_preprocessed; nnUNet_results. You should save the dataset in nnU-Net_raw. More details can be seen in [nnU-Net's dataset_format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). Then modify the environment variables.
```bash
vim ~/.bashrc
```
You should set the path to the dataset if you are using Linux.
```bash
export nnUNet_raw='data_fold_path/nnUNet_raw'
export nnUNet_preprocessed='data_fold_path/nnUNet_preprocessed'
export nnUNet_results='data_fold_path/nnUNet_results'
```
Then update the environment variables.
```bash
source ~/.bashrc
```
If you want to reproduce the Inception nnU-Net on MSWAL, please modify epoch to 1500 and initial learning rate to 0.001 in [nnUNetTrainer.py](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)(OPTIONAL).
Now you can preprocess the dataset.
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity -c 3d_fullres -p nnUNetResEncUNetLPlans
```
## Step 3: install dynamic_network_architectures
```bash
git clone https://anonymous.4open.science/r/MSWAL--406B.git
cd dynamic_network_architectures
pip install -e .
```

## Step 4: train the model.
Before training the model, you should modify network_class_name in 3d_fullres to "dynamic_network_architectures.architectures.inception.InceptionNnunet" in nnUNetResEncUNetLPlans.json and change the relevant batch_size to smaller (dependent on your GPU). Then run the training process. We use  five-fold cross-validation to train the model.
```bash
nnUNetv2_train DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetLPlans
nnUNetv2_train DATASET_ID 3d_fullres 1 -p nnUNetResEncUNetLPlans
nnUNetv2_train DATASET_ID 3d_fullres 2 -p nnUNetResEncUNetLPlans
nnUNetv2_train DATASET_ID 3d_fullres 3 -p nnUNetResEncUNetLPlans
nnUNetv2_train DATASET_ID 3d_fullres 4 -p nnUNetResEncUNetLPlans
```
## Step 5: run inference
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c 3d_fullres -f 0 1 2 3 4 -p nnUNetResEncUNetLPlans
```
## Step 6: evaluation
```bash
nnUNetv2_evaluate_folder -djfile path/dataset.json -pfile path/plans.json path/labelsTs path/infersTs
```











