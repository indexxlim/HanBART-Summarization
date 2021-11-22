# HanBART
HanBART는 BART 모델을 한국어와 영어 코퍼스를 이용하여 사전학습한 모델입니다.
Vocabulary는 자체 개발한 moran tokenizer에서 추출된 token을 포함하여 54000개의 sub-word token으로 이루어져있습니다.
사전학습에 사용된 corpus는 55G를 사용하였으며,
[**BART**](https://arxiv.org/pdf/1910.13461.pdf)(**B**idirectional and **A**uto-**R**egressive **T**ransformers)원문에서 사용하던 함수를 변경하여 3개를 사용했습니다.
사용한 노이즈 함수는 whole word masking, whole word deletion, token infilling입니다.

![nosing](./imgs/Noising_Function.png)

pytorch를 이용하여 으로 사전학습을 진행하였습니다.

# Data
사용한 데이터는  AI-Hub에 있는 문서요약 텍스트의 신문기사 1.3G의 데이터를 사용했습니다.
train set - 271093건
valid set - 30122건

# Tokenizer
학습을 위해 hanbert tokenizer를 설치합니다. 현재 tokenizer는 ubuntu환경에서만 동작합니다.
```
$> pip install hanbert_tokenizer
```

```
>>> from hanbert_tokenizer import HanBert_Tokenizer
>>> tokenizer = HanBert_Tokenizer()
>>> tokenizer
PreTrainedTokenizer(name_or_path='/home/jisu/Hanbert_tokenizer/hanbert_tokenizer/moran', vocab_size=53999, model_max_len=1000000000000000019884624838656, is_fast=False, padding_side='right', special_tokens={'bos_token': '[CLS]', 'eos_token': '[EOS]', 'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
>>> 
```

# 학습
학습을 위한 파라미터들은 configuration/train.yml 파일에 저장되어 있습니다. 
```
python run_finetuning.py
```


