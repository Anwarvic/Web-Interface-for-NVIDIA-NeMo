# Web Interface for NVIDIA NeMo

NeMo (Neural Modules) is a toolkit for creating AI applications using neural modules - conceptual blocks of neural networks that take typed inputs and produce typed outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.

NeMo makes it easy to combine and re-use these building blocks while providing a level of semantic correctness checking via its neural type system. As long as two modules have compatible inputs and outputs, it is legal to chain them together.

NeMo's API is designed to be framework-agnostic, but currently only PyTorch is supported.

The toolkit comes with extendable collections of pre-built modules for automatic speech recognition (ASR), natural language processing (NLP) and text synthesis (TTS). Furthermore, NeMo provides built-in support for distributed training and mixed precision on the latest NVIDIA GPUs.

NeMo consists of:

- NeMo Core: fundamental building blocks for all neural models and type system.
- NeMo collections: pre-built neural modules for particular domains such as automatic speech recognition (nemo_asr), natural language processing (nemo_nlp) and text synthesis (nemo_tts).


## Requirements

- python 3.6 or python 3.7
- PyTorch 1.2 or 1.3, you can install it from [here](https://pytorch.org/)
- (optional for best performance on training/evaluating models) NVIDIA [APEX](https://github.com/NVIDIA/apex).


## Install

You can install all the dependencies of this repo using only the following command:
```
pip install -r requirements.txt
```

If you want to use a language model when decoding, you need to install [Baidu's CTC decoders](https://github.com/PaddlePaddle/DeepSpeech) by running this command:
```
cd NeMo && ./scripts/install_decoders.sh && cd ..
```
