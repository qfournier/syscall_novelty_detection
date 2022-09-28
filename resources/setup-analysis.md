# Deploy the Analysis Server Locally

1. Go to the home directory:
    ```bash
    cd /home/$USER
    ```

2. Download, install, and update [Anaconda](https://www.anaconda.com):
    ```bash
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    bash Anaconda3-2019.10-Linux-x86_64.sh -b
    conda init
    conda config --set auto_activate_base false
    conda update -y conda
    conda update -y conda-build
    ```

3. Create an environment with the required packages: 
    ```bash
    conda create -y --name py3 python=3.7
    conda activate py3
    conda install -y numpy matplotlib scikit-learn seaborn nltk psutil sentencepiece pyarrow cudatoolkit=10.0
    conda install -y pytorch=1.7.1=cuda100py37h50b9e00_1 -c conda-forge
    pip install git+https://github.com/qfournier/longformer
    pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
    ```

4. (Optional) Install the dependencies for [Babeltrace](https://babeltrace.org): 
    ```bash
    sudo apt-get update
    sudo apt-get install git uuid-dev automake autoconf libtool bison flex swig asciidoc libpopt-dev libgtk2.0-dev
    ```

5. Specify the Python paths: 
    ```bash
    echo "export PYTHON_CONFIG=/home/$USER/anaconda3/envs/py3/bin/python3.7-config" >> /home/$USER/.bashrc
    echo "export PYTHON=/home/$USER/anaconda3/envs/py3/bin/python" >> /home/$USER/.bashrc
    echo "export PYTHON_PREFIX=/home/$USER/anaconda3/envs/py3/" >> /home/$USER/.bashrc
    source /home/$USER/.bashrc
    ```

6. Download, configure, build, and install Babeltrace 2: 
    ```bash
    git clone https://github.com/efficios/babeltrace.git
    cd babeltrace
    git checkout stable-2.0
    ./bootstrap
    ./configure PYTHON_LDFLAGS=-fno-lto --prefix /home/$USER/babeltrace --enable-python-bindings --enable-python-plugins --disable-debug-info
    make
    make install
    ```

7. Activate the environment and install the Python bindings:
    ```bash
    conda activate py3
    cd /home/$USER/babeltrace/src/bindings/python/bt2
    python setup.py install
    ```

8. Clean up:
    ```
    cd /home/$USER
    rm -rf Anaconda3-2019.10-Linux-x86_64.sh lib/ bin/ include/ share/ babeltrace
    ```