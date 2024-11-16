# BERT-DPT for Dialogue Systems

This repository contains the implementation of BERT-DPT using Ubuntu Dialogue Corpus V2.

## Prerequisites
This model is compatible with Python 3.12 and torch 2.6.0.dev20240923+cu121

Before you begin, ensure you have the following libraries installed:

```
six
numpy
h5py
tensorboard
evaluate
rouge_score
torch
torchvision
torchaudio
```

You can install these using pip:

```python
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 
pip install numpy six h5py tensorboard evaluate rouge_score
```

## Data Processing

1. Download the [Ubuntu Dialogue Corpus](https://drive.google.com/drive/folders/1cm1v3njWPxG5-XhEUpGH25TMncaPR7OM?usp=sharing). Place the files in the Ubuntu_Corpus_V2 folder.

2. Process the data by running:

   ```bash
   python data/data_utils.py
   ```
   After processing, you should have the following files:
   - [ubuntu_post_training.txt](https://drive.google.com/file/d/16yBz9NtJSTmXabY89N_ZdtaAGbZlZxcZ/view?usp=sharing)
   - [ubunutu_train.pkl](https://drive.google.com/file/d/1YPinTNhkZKXsgFiVcdrrHVciFmbSeSKY/view?usp=drive_link)
   - [ubuntu_valid.pkl](https://drive.google.com/file/d/1EWQfOq-ej8ArPiXxILppuQqKbuitYeJy/view?usp=sharing)
   - [ubuntu_test.pkl](https://drive.google.com/file/d/19DyD3NP1x2x-NgCgzOCFt7yIiRtHKUa3/view?usp=sharing)

3. Create BERT post-training data:

   ```bash
   python data/create_bert_post_training_data.py
   ```
   After processing, you should have the following files:
   - [ubuntu_post_training.hdf5](https://drive.google.com/file/d/14IHvS5mqsEUOMMX7tz0MVoYPWZ7NTJIR/view?usp=drive_link)
 

## Training Pipeline

### Post-Training

1. Download and place the [bert-base-uncased-pytorch_model.bin](https://drive.google.com/file/d/17mUrNowFa-833vgzLwO5JfC3lAPbBNhy/view?usp=sharing) file in `resources/bert-base-uncased/bert-base-uncased-pytorch_model.bin`

2. Run the following command:

```bash
python main.py --model bert_ubuntu_pt --train_type post_training --bert_pretrained bert-base-uncased --data_dir ./data/Ubuntu_Corpus_V2/ubuntu_post_training.hdf5
```
This will create checkpoints in `results/bert_ubuntu_pt/post_training/{TIMESTAMP}/checkpoints/` folder. Rename it to `bert-post-uncased-pytorch_model.pth` and place it in the in the `resources/bert-post-uncased` folder. 

You may skip this step by placing [bert-post-uncased-pytorch_model.pth](https://drive.google.com/file/d/1VY9MpLJz6Zxe3KiCQ5fUmH7g8Bra-lbp/view?usp=sharing) in the `resources/bert-post-uncased` folder.

### Fine-Tuning

1. Run the following command:

```bash
python main.py --model bert_dpt_ft --train_type fine_tuning --bert_pretrained bert-post-uncased
```
This will create checkpoints in `results/bert_dpt_ft/fine_tuning/{TIMESTAMP}/checkpoints/` folder. Place the final checkpoint in the `results/bert_dpt_ft/fine_tuning/` folder.

You may skip this step by placing [checkpoint.pth](https://drive.google.com/file/d/1qV2g8RoCtu2DAnAcAiom9Mh-RvaFaOYE/view?usp=drive_link) in the `results/bert_dpt_ft/fine_tuning` folder.

## Evaluation

For evaluation, run:

```bash
python  main.py --model bert_dpt_ft --train_type fine_tuning --bert_pretrained bert-post-uncased --evaluate results/bert_dpt_ft/fine_tuning/[TIMESTAMP]/checkpoints/checkpoint.pth
```

This will print evaluation metrics and generate prediction scores

## Calculate ROUGE Score

To calculate the ROUGE score, execute:

```bash
python compute_rouge.py
```

## Log Files

We have shared log files from our experiments in the `Log Files` folder

## Acknowledgements
[BERT-DPT GitHub Repository](https://github.com/taesunwhang/BERT-ResSel) \
[An Effective Domain Adaptive Post-Training Method for BERT in Response Selection](https://arxiv.org/abs/1908.04812v2)
