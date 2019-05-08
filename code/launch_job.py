import os

slurm_template = """#!/bin/bash
#SBATCH -t 1-23:59
#SBATCH --mem=8000
#SBATCH -p doshi-velez
#SBATCH -o log.out
#SBATCH -e err.out

module load Anaconda3/5.0.1-fasrc02
source activate jzdu
python icu_dops.py {}
"""

def launch_job(prefix='', drops=int, test_size=0.01, batch_size=32):
    savepath = 'result/' + prefix
    savepath += 'drops-' + str(drops)
    savepath += '__test_size-' + str(test_size)
    savepath += '__batch_size-' + str(batch_size)
    args = "--savepath={} --drops={} --test_size={} --batch_size={}"\
            .format(savepath, drops, test_size, batch_size)
    slurm = slurm_template.format(args)
    with open("tmpfile.slurm", "w") as f:
        f.write(slurm)
    os.system("cat tmpfile.slurm | sbatch")
    os.system("rm tmpfile.slurm")

# exclude basic measure: hr, meanbp, urine, temp, map
for i in range(3,12):
    for j in range(1,11):
        launch_job(prefix='quantile__', drops=i+1, test_size=j/100, batch_size=32)