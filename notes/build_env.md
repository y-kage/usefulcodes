# Prepare
- Version
  - python & CUDA \
    Check supported version of NVIDIA Driver for the GPU. See [here](https://www.nvidia.co.jp/Download/index.aspx?lang=jp). \
    Check supported CUDA version for the NVIDIA Driver. See [here](https://developer.nvidia.com/cuda-gpus). \
    Check supported version of PyTorch( or TensorFlow).
    - [PyTorch](https://github.com/pytorch/pytorch/blob/master/RELEASE.md) \
      (Example) \
      <img width="452" alt="image" src="https://github.com/y-kage/usefulcodes/assets/125951749/82fbc561-f29b-4c47-bb2e-cbd9cd931359">
    - [TensorFlow](https://www.tensorflow.org/install?hl=ja) \
      (Example) \
      <img width="361" alt="image" src="https://github.com/y-kage/usefulcodes/assets/125951749/aa7bf2a4-c30a-4581-84b9-edb4669cdcc3">
    - Command for knowing GPU compute capability
      ```bash
      nvidia-smi --query-gpu=compute_cap --format=csv
      ```
  - NVIDIA Driver \
    NVIDIA Driver seems to be install with CUDA. After installing CUDA, run the command below. \
    ```$ lsmod | grep nouveau``` \
    If any detail came out, Nouveau should be disabled. See [here](https://k-hyoda.hatenablog.com/entry/2020/07/09/223907).
    
- Remove CUDA (If needed)\
  Check /usr/local/. If directory including 'cuda' in the name does not exist, cuda is installed. If you want to remove, run the commands below. See [here](https://qiita.com/harmegiddo/items/86b295ccf96eff489e02). When multiple version existing, change it using ```sudo update-alternatives --config cuda```. See [here](https://qiita.com/ketaro-m/items/4de2bd3101bcb6a6b668).
  ```
  $ sudo apt-get --purge remove nvidia*
  $ sudo apt-get --purge remove cuda*
  $ sudo apt-get --purge remove cudnn*
  $ sudo apt-get --purge remove libnvidia*
  $ sudo apt-get --purge remove libcuda*
  $ sudo apt-get --purge remove libcudnn*
  $ sudo apt-get autoremove
  $ sudo apt-get autoclean
  $ sudo apt-get update
  $ sudo rm -rf /usr/local/cuda*
  ```

# Install
- NVIDIA Driver
  See [here](https://qiita.com/porizou1/items/74d8264d6381ee2941bd)
- CUDA \
  Run the command below to get information of nvidia driver in advance.
  ```bash
  $ sudo add-apt-repository ppa:graphics-drivers/ppa
  ```
  Find the selected version from [here](https://developer.nvidia.com/cuda-toolkit-archive).
  Run commands to check your platform.
  - OS, Architecture \
    ```$ uname –a```
  - Distribution, Version \
    ```$ cat /etc/*release```

  Follow the instructions and run the commands.

- PyTorch \
  See [here](https://pytorch.org/) for latest or [here](https://pytorch.org/get-started/previous-versions/) for previous versions. \
  Select the category according to your env, and run the command.
- TensorFlow \
  Install by pip.

# Check Installation
- Nvidia Driver \
  CUDA version shown is the max supported version by the driver. \
  ```$ nvidia-smi```
- CUDA \
  PATH may not be configured. If so, set PATH or check the directory below /usr/local/. \
  ```$ nvcc -V```
- PyTorch and TensorFlow
  ```python
  import torch
  import tensorflow as tf

  print(torch.__version__)
  print(tf.__version__)
  ```

# Virtual Environment(VE)
  Make VE to protect local env and to easily change env.
- Docker \
  Be carefull to port number.
- venv \
  See [here](https://zerofromlight.com/blogs/detail/4/) if not installed
- pyenv
- Anaconda \
  Be careful pip. Run ```$ conda install pip``` in advance. See [here](https://qiita.com/en3/items/99de0098ec5668070f75).


# Reference
- [About Version](https://qiita.com/konzo_/items/a6f2e8818e5e8fcdb896)
- [About Install](https://qiita.com/konzo_/items/3e2d1d7480f7ef632603)
