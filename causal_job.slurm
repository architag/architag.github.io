#!/bin/bash
#SBATCH --job-name=ddp2
#SBATCH --account=csci_ga_3033_077-2025sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay  /scratch/sk12184/overlay-25GB-500K.ext3:rw /scratch/sk12184/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate bdml_env
cd /home/sk12184/projects/bdml-final-project
python casual_graph.py --data_dir ../BDML-Assignment1/climate_text_dataset_processed/eval
"