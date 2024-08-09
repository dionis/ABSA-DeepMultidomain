# Lifelong Learning of Aspects (LLA)

> Aspect Based Sentiment Analysis, PyTorch Implementations.
>
*  a hybrid model that combines the continual and deep learning approaches for Aspect Sentiment Clasification.

   First, a text preprocess module extracts the
   aspect word candidates (i.e., noun, adverbs) and the
   proposed model classiϐies each aspect into one of
   three possible classes: positive, negative, or neutral.

   The model starts from a Bidirectional Encoder 
   Representations from Transformers (BERT) model and
   improves the Continual Learning (CL) disadvantages based on:

   - Combining a CL regularization approach in NLP (i.e.,
ABSA) with a gradient descent modiϐication algorithm to preserve relevant weights in a CL scenario.

   - Using the output of a pretrained BERT model to
improve the results and tune the BERT model on the
CL process.

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

## Requirement

* torch >= 1.3.0
* numpy >= 1.13.3
* scikit-learn >= 0.20.4
* sklearn
* psutil >= 5.4.8
* transformers >= 2.11.0
* pytorch_transformers >= 1.2.0
* python 3.6 / 3.7

* python 3.6 / 3.7
* pytorch-pretrained-bert 0.6.1
  * See [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) for more detail.
  Note: BERT multilingual name "bert-base-multilingual-uncased"
## Usage

### Training

```sh
python train.py --model_name bert_spc --dataset restaurant
python train.py --model_name mgan --dataset twitter
```

See [train.py](./train.py) for more training arguments.

Refer to [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) for k-fold cross validation support.

### Inference

Please refer to [infer_example.py](./infer_example.py).

### Tips

* For non-BERT-based models, training procedure is not very stable.
* BERT-based models are more sensitive to hyperparameters (especially learning rate) on small data sets, see [this issue](https://github.com/songyouwei/ABSA-PyTorch/issues/27).
* Fine-tuning on the specific task is necessary for releasing the true power of BERT.

## Notes

Dionis López's Phd research with: sources code ([LifelongABSA.py](./models/LifelongABSA.py)) for training a deep and lifelong model and article ["A model of continual and deep learning for aspect based in sentiment analysis"](https://sciendo.com/pdf/10.14313/jamris/1-2023/1), Journal of Automation, Mobile Robotics and Intelligent Systems 17 (1), 3-12.

Thesis and Phd on [documentation](./documentation/) directory:

- Phd dissertation (Spanish) [defensaPhD.pdf](./documentation/defensaPhD.pdf)

- Phd Thesis (Spanish) [defensaPhD.pdf](./documentation/Thesis.pdf)

- Paper in Journal publication ["A model of continual and deep learning for aspect based in sentiment analysis"](./documentation/A_model_of_continual_and_deep_learning_for_aspect.pdf.pdf)

There are other models for training evaluation results, such as:



### AEN-BERT ([aen.py](./models/aen.py))
Song, Youwei, et al. "Attentional Encoder Network for Targeted Sentiment Classification." arXiv preprint arXiv:1902.09314 (2019). [[pdf]](https://arxiv.org/pdf/1902.09314.pdf)

![aen](assets/aen.png)

### BERT for Sentence Pair Classification ([bert_spc.py](./models/bert_spc.py))
Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018). [[pdf]](https://arxiv.org/pdf/1810.04805.pdf)

![bert_spc](assets/bert_spc.png)


## Non-BERT-based models

### MGAN ([mgan.py](./models/mgan.py))
Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)

![mgan](assets/mgan.png)

### AOA ([aoa.py](./models/aoa.py))
Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)

![aoa](assets/aoa.png)

### TNet ([tnet_lf.py](./models/tnet_lf.py))
Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)

![tnet_lf](assets/tnet_lf.png)

### Cabasc ([cabasc.py](./models/cabasc.py))
Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

![cabasc](assets/cabasc.png)


### RAM ([ram.py](./models/ram.py))
Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)

![ram](assets/ram.png)


### MemNet ([memnet.py](./models/memnet.py))
Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)

![memnet](assets/memnet.png)


### IAN ([ian.py](./models/ian.py))
Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)

![han](assets/han.png)

### ATAE-LSTM ([atae_lstm.py](./models/atae_lstm.py))
Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.

![han](assets/atae-lstm.png)


### TD-LSTM ([td_lstm.py](./models/td_lstm.py))

Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)

![td-lstm](assets/td-lstm.png)


### LSTM ([lstm.py](./models/lstm.py))

![lstm](assets/lstm.png)


## Reviews / Surveys

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)

Biesialska M., et al. “Continual lifelong learning in natural language processing: A survey”, Proceedings of the 28th
International Conference on Computational Linguistics, 2020, pp. 6523–6541 [[pdf]](https://upcommons.upc.edu/bitstream/handle/2117/341126/Continual?sequence=3)

## Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

## Licence

MIT License
