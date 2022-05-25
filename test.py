import torch
from transformers import BertJapaneseTokenizer, BertModel
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

"""
text = '明日の天気は晴れだ。'
encoding = tokenizer(
    text,max_length=12, padding='max_length', truncation=True
)
print(encoding)
"""

# 4-14
# モデルのロード
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
bert = BertModel.from_pretrained(model_name)

# BERTをGPUに載せる
bert = bert.cuda() 
# 4-16
text_list = [
    '明日は自然言語処理の勉強をしよう。',
    '明日はマシーンラーニングの勉強をしよう。'
]

# 文章の符号化
encoding = tokenizer(
    text_list,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print(type(encoding))

# データをGPUに載せる
encoding = { k: v.cuda() for k, v in encoding.items() } 

print(encoding)

# BERTでの処理
output = bert(**encoding) # それぞれの入力は2次元のtorch.Tensor
last_hidden_state = output.last_hidden_state # 最終層の出力