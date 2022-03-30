env_dir=env

module unload python_cpu
#module load -q gcc/6.3.0
module load python_gpu/3.7.4
module load eth_proxy
module load hdf5/1.10.1

if [ ! -d "$env_dir" ]; then
    python3 -m pip install --user virtualenv
    python3 -m virtualenv \
        --system-site-packages \
        "$env_dir"
fi

source "$env_dir/bin/activate"
