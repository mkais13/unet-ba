#!/bin/bash

#SBATCH --export=NONE

#SBATCH --nodes=1

#SBATCH --ntasks-per-node={tasks_per_node}

#SBATCH --mem={mem}G

#SBATCH --gres=gpu:1

#SBATCH --partition={partition}

#SBATCH --time={time}

#SBATCH --job-name=kh03_{i}_{log_name}

#SBATCH --output=/scratch/tmp/m_kais13/slogs/{log_name}.log

#SBATCH --mail-type=ALL

#SBATCH --mail-user=moritz.kaiser@uni-muenster.de

ml palma/2019a
ml fosscuda/2019a
ml Singularity
cd $HOME/unet
singularity exec tensorflow:1.10.1-devel-gpu-py3.sif python {python_script}



