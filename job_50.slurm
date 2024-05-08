#!/bin/bash
#SBATCH --job-name=SAGAN         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=ka3866@princeton.edu

module purge
module load anaconda3/2024.2
conda activate segan-leftthomas

TRIAL="sagan-50b-4hr-DSEdinburgh5000-t1"
BATCH_SIZE=50
NUM_EPOCHS=100
DATASET="DS_10283_2791_subset"

python main.py -t $TRIAL -b $BATCH_SIZE -e $NUM_EPOCHS -d $DATASET

mv "./results/$TRIAL/loss_train.txt" "./results/$TRIAL/loss_train_$TRIAL.txt"
rclone copy "./results/$TRIAL/loss_train_$TRIAL.txt" dropbox:results

tar -czvf "./results/$TRIAL/$TRIAL-outputaudio.tar.xz" *wav*
rclone copy "./results/$TRIAL/$TRIAL-outputaudio.tar.xz" dropbox:results
# rm "./results/$TRIAL/$TRIAL-outputaudio.tar.xz"
rm ./results/$TRIAL/*wav*

tar -czvf "./epochs/$TRIAL/$TRIAL-checkpoints.tar.xz" *pkl*
rclone copy "./epochs/$TRIAL/$TRIAL-checkpoints.tar.xz" dropbox:results
# rm "./epochs/$TRIAL/$TRIAL-checkpoints.tar.xz"
rm ./epochs/$TRIAL/*pkl*