# Can BERT Refrain from Forgetting on Sequential Tasks? A Probing Study
Open-resource code of our ICLR 2023 paper:
**Can BERT Refrain from Forgetting on Sequential Tasks? A Probing Study**
[https://openreview.net/forum?id=UazgYBMS9-W](https://openreview.net/forum?id=UazgYBMS9-W)

---

## Requirements
python==3.8
pytorch>=1.7.0
transformers>=4.3.0

---

## Download Datasets
We follow d'Autume et al. (2019), and download the datasets from [their Google Drive](http://goo.gl/JyCnZq).

---

## Prepare

1. Clone from Github.
2. Create the data directory:
```bash
cd plms_are_lifelong_learners
mkdir data
```

3. Move the .tar.gz files to "data".
4. Uncompress the .tar.gz files and sample data from the original datasets:
```bash
bash uncompressing.sh
python sampling_data.py --seed 42
```

---

## Train Models Sequentially

1. Tokenize the input texts:
```bash
python tokenizing.py --tokenizer bert-base-uncased --data_dir ./data/ --max_token_num 128
```
"bert-base-uncased" can be replaced by any other tokenizer in [Hugging Face Transformer Models](https://huggingface.co/models), e.g. "roberta-base", "prajjwal1/bert-tiny", etc.
The files with tokenized texts will be saved in the directory ./data/

2. Train the models 
```bash
CUDA_VISIBLE_DEVICES=0 python train_cla.py --plm_name bert-base-uncased --tok_name bert-base-uncased --pad_token 0 --plm_type bert --hidden_size 768 --device cuda --seed 1023 --padding_len 128 --batch_size 32 --learning_rate 1.5e-5 --trainer sequential --epoch 2 --order 0 --rep_itv 10000 --rep_num 100
```

- "plm_name" indicates which pre-trained model (and its well-pre-trained weights) should be loaded. "tok_name" means which tokenizer are employed in the first step. These two parameters can be different, e.g., python train_cla.py --plm_name bert-large-uncased --tok_name bert-base-uncased ...
- "plm_type" is required. When using GPT-2 or XLNet, please set "plm_type" as "gpt2" or "xlnet". It is because the classifiaction models based on GPT-2 or XLNet employ representations of the last tokens as features, while other models (like BERT or RoBERTa) employ the first token ([CLS]).
- "pad_token" should be an integer, which is the ID of padding tokens in the tokenizer, e.g. 0 for BERT, or 1 for RoBERTa.
- "trainer" can be _sequential_, _replay_, or _multiclass_. The _sequential_ means training sequentially without Episodic Memory Play, and _multiclass_ means training on all tasks together (multi-task learning).
- "order" can be 0, 1, 2, 3, which is correspond to Appendix A in the paper.

---

## Probing Study

1. Re-train the decoder in each checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python probing_train.py --plm_name bert-base-uncased --tok_name bert-base-uncased --pad_token 0 --plm_type bert --hidden_size 768 --device cuda --seed 1023 --padding_len 128 --batch_size 32 --learning_rate 3e-5 --epoch 10 --train_time "1971-02-03-14-56-07"
```

2. Evaluate the performance of each re-trained model:
```bash
CUDA_VISIBLE_DEVICES=0 python test_model.py --plm_name bert-base-uncased --tok_name bert-base-uncased --pad_token 0 --plm_type bert --hidden_size 768 --device cuda --padding_len 128 --train_time "1971-02-03-14-56-07"
```
