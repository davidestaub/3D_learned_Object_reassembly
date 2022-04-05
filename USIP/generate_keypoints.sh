bsub -R "rusage[ngpus_excl_p=1,mem=5000]" python ./USIP/evaluation/save_keypoints.py
