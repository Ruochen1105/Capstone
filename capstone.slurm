#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=200G
#SBATCH --output=/gpfsnyu/home/rm5327/output/%j.out
#SBATCH --err=/gpfsnyu/home/rm5327/err/biased_mnist.%j.err
#SBATCH --partition=aquila
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2080Ti:8

module purge
module load miniconda
source activate capstone

date
python < capstone.py
date