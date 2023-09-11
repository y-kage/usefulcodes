# Virtual Environment(VE)
  Make VE to protect local env and to easily change env
- Docker \
  Be carefull to port number.
- venv
- pyenv
- Anaconda \
  Be careful pip. Run ```$ conda install pip``` in advance. See [here](https://qiita.com/en3/items/99de0098ec5668070f75).

# Check list
- Version
  - NVIDIA Driver
  - python & CUDA \
    Check supported version of PyTorch( and TensorFlow).
    - [PyTorch](https://github.com/pytorch/pytorch/blob/master/RELEASE.md) \
      (Example) \
      <img width="452" alt="image" src="https://github.com/y-kage/usefulcodes/assets/125951749/82fbc561-f29b-4c47-bb2e-cbd9cd931359">
    - [TensorFlow](https://www.tensorflow.org/install?hl=ja) \
      (Example) \
      <img width="361" alt="image" src="https://github.com/y-kage/usefulcodes/assets/125951749/aa7bf2a4-c30a-4581-84b9-edb4669cdcc3">

    
- CUDA \
  Check /usr/local/. If directory including 'cuda' in the name does not exist, cuda is installed. If you want to remove, run the commands below. See [here](https://qiita.com/harmegiddo/items/86b295ccf96eff489e02).
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

- 

# Install

# Reference
- [About Version](https://qiita.com/konzo_/items/a6f2e8818e5e8fcdb896)
- [About Install](https://qiita.com/konzo_/items/3e2d1d7480f7ef632603)
