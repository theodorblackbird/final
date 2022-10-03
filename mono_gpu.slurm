#!/bin/bash
#SBATCH -C v100-32g
#SBATCH --job-name=gpu_mono          # nom du job
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=gpu_mono%j.out      # nom du fichier de sortie
#SBATCH --error=gpu_mono%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
# --qos=qos_gpu-t4  
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load tensorflow-gpu/py3/2.8.0
#module load pytorch-gpu/py3/1.10.0
module load libsndfile
 
# echo des commandes lancees
set -x
 
# execution du code
python -u train.py #--continue_checkpoint 165 20220807-032943
