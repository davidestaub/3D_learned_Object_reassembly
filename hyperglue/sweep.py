import argparse
import os
import subprocess
import time

if __name__ == '__main__':
    for run in range(5):
        subprocess.check_call(
            [
                "bsub",
                "-n 4 -R \"rusage[ngpus_excl_p=1, mem=5000]\" -R \"select[gpu_mtotal0>=10240]\" "
                "wandb agent matvogel/3D_learned_Object_reassembly-hyperglue/fj91yu0g",
            ]
        )
