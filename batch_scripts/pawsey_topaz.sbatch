#!/bin/bash --login
#SBATCH --partition=nvlinkq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --account=y95
#SBATCH --export=NONE

unset PYTHONPATH

#Default loaded compiler module is gcc module
module unload cuda
module unload gcc

module load gcc/11.1.0
module load cuda/11.4.2

source ~/.bashrc
conda activate ./gperceiver/condaenv

LD_LIBRARY_PATH="${PWD}/gperceiver/condaenv/lib/:${LD_LIBRARY_PATH:-}"

mkdir -p last_allele best_allele

gperceiver_pretrain \
  --model ./pretrain_allele \
  --nepochs 200 \
  --batch-size 16 \
  --contrastive-allele-weight 0.0 \
  --last-checkpoint last_allele/chk \
  --best-checkpoint best_allele/chk \
  --reduce-lr-patience 5 \
  --early-stopping-patience 20 \
  --prop-x 0.5 \
  --loss binary_focal \
  --logs logs_allele.txt \
  chroms.tsv combined_genos.tsv
