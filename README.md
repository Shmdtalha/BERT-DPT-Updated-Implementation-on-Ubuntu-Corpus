# BERT-DPT Updated Implementation on Ubuntu Corpus
This repository contains an updated implementation of An Effective Domain Adaptive Post-Training Method for BERT in Response Selection (BERT-DPT). This implementation will work with the latest version of Python.

## Setting up the Environment
An A100 GPU (about 7 instances) was used to speed up training since BERT-DPT is very resource-intensive. It is available on Colab Pro, but you may work without it. Please note that it will take far longer with other architectures.

## Download the bin file
You may download the file here: [bert-base-uncased-pytorch_model.bin](https://drive.google.com/file/d/1BEonF_eclgLSsfD-xJN8mJqzYy31bGNK/view?usp=sharing) \
Place it in this directory: resources/bert-base-uncased

## Dataset
You may download the dataset here: [Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing) 

## Preprocessing
Preprocessing returns a `ubuntu_post_training.hdf5` file, which must be placed in data/Ubuntu_Corpus_V1 directory
```python
!python data/create_bert_post_training_data.py
```
## Domain Post Training
```python
!python re_main.py --model bert_ubuntu_pt --train_type post_training --bert_pretrained bert-base-uncased --data_dir /data/Ubuntu_Corpus_V1/ubuntu_post_training.hdf5
```
## Fine Tuning
Fine tuning was performed using BERT base
```python
!python main.py --model bert_base_ft --train_type fine_tuning --bert_pretrained bert-base-uncased
```
## Evaluation
To evaluate your model and obtain checkpoints, run the line below
```python
!python re_main.py --model bert_dpt_ft --train_type fine_tuning --bert_pretrained bert-post-uncased --evaluate results/checkpoints/checkpoint_.pth
```
## Acknowledgements
[BERT-DPT GitHub Repository](https://github.com/taesunwhang/BERT-ResSel)
[An Effective Domain Adaptive Post-Training Method for BERT in Response Selection](https://arxiv.org/abs/1908.04812v2)
