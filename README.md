# Web Interface for NVIDIA NeMo

This repo is an attempt to combine all three main components of NeMo into a web interface that can be used easily without the need to dig deeper into the toolkit itself. 


## NeMo Overview

NeMo stands for "Neural Modules" and it's a toolkit with a collections of pre-built modules for automatic speech recognition (ASR), natural language processing (NLP) and text synthesis (TTS). NeMo consists of:

- NeMo Core: contains building blocks for all neural models and type system.
- NeMo collections: contains pre-built neural modules for ASR, NLP and TTS.

NeMo's is designed to be framework-agnostic, but currently only PyTorch is supported. Furthermore, NeMo provides built-in support for distributed training and mixed precision on the latest NVIDIA GPUs.


## Installation

To get started with this repository, you need to install:

- PyTorch 1.2 or 1.3 from [here](https://pytorch.org/).
- Clone the [NeMo](https://github.com/NVIDIA/NeMo) repository on GitHub:
    ```
    git clone --depth 1 https://github.com/NVIDIA/NeMo.git
    ```
- Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### GPU Support (OPTIONAL)

If your machine supports Cuda, then you need to install NVIDIA [APEX](https://github.com/NVIDIA/apex) for best performance on training/evaluating models.
```
cd NeMo
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Using Language Model with ASR (OPTIONAL)

If you want to use a language model when using ASR model, then you need to install [Baidu's CTC decoders](https://github.com/PaddlePaddle/DeepSpeech):
```
cd NeMo
./scripts/install_decoders.sh
cd ..
```
