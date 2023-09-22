---

#  Training LLMs models with Quantization and PEFT

This repository contains different code train LLMs using different datasets with quantization and PEFT (Progressive Embedding Fine-Tuning). 

For now, there is only one script `stf_train_opt350m.py` for finetuning `facebook/opt350m` model on the codealpaca20k dataset. The current setup is focused on using a very small CPU with at most 12Gb.

## Dependencies

- torch
- datasets
- peft
- transformers
- trl
- pynvml

## Overview

 code performs the following steps:

1. **Configuration for Quantization**: Sets up the Bits and Bytes configuration for quantizing the model.
2. **Model Loading**: Loads a pretrained model (`facebook/opt-350m`) and its tokenizer.
3. **PEFT Configuration**: Configures the model for Progressive Embedding Fine-Tuning (PEFT).
4. **Training Configuration**: Sets up the training arguments.
5. **Dataset Loading**: Loads the CodeAlpaca dataset.
6. **Data Formatting**: Formats the dataset for training.
7. **Training**: Trains the model using the SFTTrainer.
8. **Saving the Model**: Saves the trained model.

## Usage

To run the code, simply execute the provided script. Ensure that all dependencies are installed and the dataset is accessible.

```bash
python stf_train_opt350m.py
```
## Notes

- The model is trained using quantization to reduce memory usage and improve speed.
- PEFT (Progressive Embedding Fine-Tuning) is used to fine-tune the model.
- The dataset used is `lucasmccabe-lmi/CodeAlpaca-20k`.

---
