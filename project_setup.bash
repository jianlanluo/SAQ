# Project setup script
# Source this file to set up the environment for this project.


if [ ! -f ./environment.yml ]; then
    exit -1
fi
ENV_NAME=$(cat ./environment.yml | egrep "name: .+$" | sed -e 's/^name:[ \t]*//')

if [ "$1" = "setup" ]; then
    echo "Creating conda environment..."
    export CONDA_OVERRIDE_CUDA="11.3"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin"
    conda env create -f environment.yml
elif [ "$1" = "remove" ]; then
    conda remove --name $ENV_NAME --all --yes
elif [ "$1" = "build_base" ]; then
    rm -f containers/base_img.sif
    singularity build --fakeroot containers/base_img.sif base_container.def
elif [ "$1" = "build" ]; then
    rm -f containers/code_img.sif
    singularity build --fakeroot containers/code_img.sif code_container.def
elif [ "$1" = "build_all" ]; then
    rm -f containers/base_img.sif
    singularity build --fakeroot containers/base_img.sif base_container.def
    rm -f containers/code_img.sif
    singularity build --fakeroot containers/code_img.sif code_container.def
else
    conda activate $ENV_NAME
    export PROJECT_HOME="$(pwd)"
    alias ph="cd $PROJECT_HOME"
    alias notebook='jupyter notebook --port 5008 --ip=* --NotebookApp.allow_origin="https://colab.research.google.com" --NotebookApp.port_retries=0'

    export CONDA_OVERRIDE_CUDA="11.2"
    export XLA_PYTHON_CLIENT_PREALLOCATE='false'
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
    export WANDB_API_KEY=''

    export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/vqn"
fi
