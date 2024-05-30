# SLOOP
## File Description
> src/SLOOP.py : main code
```sh
python src/SLOOP.py
```
> data/log : save the result file of loop closure detection.

> src/analyze-benchmark.py : analyze the F1, EP, AP metrics and plot P-R curves.

> src/pose_refine_readme.md : show the usage of pose refining.
 
> src/data/icp_data : save the result file after pose refining.

## Dataset
We provide the semantic kitti 07 dataset here:   [semantic_kitti-07](https://drive.google.com/file/d/1iXjwXXzNzO5IFKGdadpnsSkPiuydwwUF/view?usp=sharing). Please uzip and save it in 'your dataset path'.

change the 'dataset_path' in <config/sk_preprocess.yaml> to 'your dataset path'.

## Result Files
/neg100 saves the result files corresponding to Ours-SK in Fig. 5.
