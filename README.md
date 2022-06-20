# 日本語版 Improving Visual-Semantic Embeddings with Hard Negatives

## 変更点
主な変更点は下記の通りです。
* 日本語への対応
* Stair Captionsデータセットの使用
* Text EncoderとしてBERTを追加し、精度を向上させた

## 検索結果の例
* 画像検索
![image](https://user-images.githubusercontent.com/3047275/174529861-d416bda8-143b-4fef-901c-1b2f0d820d61.png)
* キャプション検索
![image](https://user-images.githubusercontent.com/3047275/174529900-9c082444-b3cf-41a1-8361-0940373e5d19.png)



Code for the image-caption retrieval methods from
**[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612)**
*, F. Faghri, D. J. Fleet, J. R. Kiros, S. Fidler, Proceedings of the British Machine Vision Conference (BMVC),  2018. (BMVC Spotlight)*

## Dependencies
We recommended to use Anaconda for the following packages.

* Python 3.5
* [PyTorch](http://pytorch.org/) (>0.2)
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* [pycocotools](https://github.com/cocodataset/cocoapi)
* [torchvision]()
* [matplotlib]()


* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data

Download the dataset files and pre-trained models. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The precomputed image features are from [here](https://github.com/ryankiros/visual-semantic-embedding/) and [here](https://github.com/ivendrov/order-embedding). To use full image encoders, download the images from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

```bash
wget http://www.cs.toronto.edu/~faghri/vsepp/vocab.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar
```

We refer to the path of extracted files for `data.tar` as `$DATA_PATH` and 
files for `models.tar` as `$RUN_PATH`. Extract `vocab.tar` to `./vocab` 
directory.

*Update: The vocabulary was originally built using all sets (including test set 
captions). Please see issue #29 for details. Please consider not using test set 
captions if building up on this project.*

## Evaluate pre-trained models

```python
python -c "\
from vocab import Vocabulary
import evaluation
evaluation.evalrank('$RUN_PATH/coco_vse++/model_best.pth.tar', data_path='$DATA_PATH', split='test')"
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco`.

## Training new models
Run `train.py`:

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --logger_name 
runs/coco_vse++ --max_violation
```

Arguments used to train pre-trained models:

| Method    | Arguments |
| :-------: | :-------: |
| VSE0      | `--no_imgnorm` |
| VSE++     | `--max_violation` |
| Order0    | `--measure order --use_abs --margin .05 --learning_rate .001` |
| Order++   | `--measure order --max_violation` |


## Reference

If you found this code useful, please cite the following paper:

    @article{faghri2018vse++,
      title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
      author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
      booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
      url = {https://github.com/fartashf/vsepp},
      year={2018}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
