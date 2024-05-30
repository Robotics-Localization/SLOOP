# SLOOP
## File Description
> src/SLOOP.py : main code

Change the < dataset path >  in line 1103 of SLOOP.py before running SLOOP.py.

> data/log : save the result file of loop closure detection.

> src/analyze-benchmark.py : analyze the F1, EP, AP metrics and plot P-R curves.

> src/pose_refine_readme.md : show the usage of pose refining.
 
> src/data/icp_data : save the result file after pose refining.

## Dataset
We provide the semantic kitti 07 dataset here:   [semantic_kitti-07](https://drive.google.com/file/d/1iXjwXXzNzO5IFKGdadpnsSkPiuydwwUF/view?usp=sharing). Please uzip and save it in 'your dataset path'.

change the 'dataset_path' in <config/sk_preprocess.yaml> to 'your dataset path'.

## Result Files
/neg100 saves the result files corresponding to Ours-SK in Fig. 5.

## Run
The first time you run, you need to install some libraries:
```sh
conda create -n SLOOP python=3.9
conda activate SLOOP
conda install -c conda-forge opencv
conda install numpy matplotlib pandas
conda install -c open3d-admin open3d
pip install pyyaml==5.3.1 rospkg==1.5.0
pip install pycryptodomex
pip install gnupg
conda install -c open3d-admin -c conda-forge open3d
python src/SLOOP.py
```
Afterwards：
```sh
conda activate SLOOP
python src/SLOOP.py
```
