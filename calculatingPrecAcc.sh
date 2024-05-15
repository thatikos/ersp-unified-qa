#!/bin/bash

#SBATCH -o ./slurm_output/%A.out
#SBATCH -e ./slurm_output/%A.err
#SBATCH --partition=gypsum-2080ti
#SBATCH -G 8
#SBATCH -c 2
#SBATCH --mem=375G
#SBATCH -t 7-00:00:00


source /project/pi_hzamani_umass_edu/asalemi/ERSP/venv/bin/activate

python /project/pi_hzamani_umass_edu/asalemi/ERSP/full_pipeline/inferenceCopy.py \
    --retriever_path "/work/pi_hzamani_umass_edu/REML/multi_task/pretrained_models/nq_retriever" \
    --validation_data "/project/pi_hzamani_umass_edu/asalemi/ERSP/nq/nq_dev.json" \
    --passages_path "/work/pi_hzamani_umass_edu/REML/multi_task/psgs_w100.tsv" \
    --index_directory "/gypsum/work1/zamani/asalemi/ersp_index/temp_index" \
    --do_eval \
