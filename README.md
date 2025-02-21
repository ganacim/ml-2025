# Machine Learing Course: Code and Exercices

## References

### CMake
https://code.visualstudio.com/docs/cpp/cmake-linux#_create-a-cmake-hello-world-project

### NVidia NSigth
Download: https://developer.nvidia.com/nsight-systems/get-started
NVTX: in 01_cuda/ clone https://github.com/NVIDIA/NVTX.git
https://nvidia.github.io/NVTX/doxygen-cpp/index.html

Tutorial de otimização em Python: https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51182/?playlistId=playList-456b1087-4a35-4d04-8aa6-9af5c0b2d20f


## Python Setup

### Installing pyenv and poetry

Install pyenv with
```sh
curl https://pyenv.run | bash
```

Add
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
at the end of *~/.bashrc* to enable pyenv.

Install Poetry with
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
Add
```bash
export PATH="$HOME/.local/bin:$PATH"
```
at the end of *~/.bashrc* to enable Poetry.

Restart bash session for the changes to take effect, or run
```sh
. ~/.bashrc
```

Install Python 3.12 and activate it run
```sh
pyenv install 3.12
```
If you get any errors, your system may be missing some packages. Try to fix that by installing:
```sh
sudo apt install \
    build-essential \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev
```

To enable VSCode compatibility run the following command (this is optional but recomended)
```sh
poetry config virtualenvs.in-project true
```

Enable python 3.12 from pyenv with
```sh
pyenv shell 3.12
```

Enable poetry virtualenv with
```sh
poetry env use python
```
Using Poetry install all dependencies.
```sh
poetry install
```

Install git filters for notebooks with
```sh
poetry shell
nbstripout --install --attributes .gitattributes
```

Install pre-commit hooks with
```sh
pre-commit install
```