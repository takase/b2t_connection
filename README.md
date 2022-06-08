# On Layer Normalizations and Residual Connections in Transformers

This repository contains transformers with B2T connection proposed in our paper.
B2T connection mitigates unstable training property in Post-LN Transformers while maintaining all the advantages of Post-LN. Please check our paper for more details.

>[On Layer Normalizations and Residual Connections in Transformers](https://arxiv.org/abs/2206.00330)

>Sho Takase, Shun Kiyono, Sosuke Kobayashi, Jun Suzuki

![Method](./method.png "Methods")

As an example, this document provides the way to train the Post-LN based Transformer with B2T connection on WMT En-De.


## Requirements

- PyTorch version >= 1.4.0
- Python version >= 3.6


## WMT En-De

### Training

##### 1. Download and pre-process datasets following the description in [this page](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt)

##### 2. Train model

Run the following command on 4 GPUs.

```bash
python -u train.py \
    pre-processed-data-dir \
    --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 3584 --min-lr 1e-09 --update-freq 32  --log-interval 100  --max-update 50000 \
    --residual-bottom-to-top \
    --share-all-embeddings --keep-last-epochs 10 --seed 1 --save-dir model-save-dir
```

If you train a deeper model than 6L-6L, we recommend setting `--dropout` to 3.5 (or 4.0) to achieve better performance.
In addition, using perturbations such as adding `--sampling-method worddrop --enc-replace-rate 0.1 --dec-replace-rate 0.1` probably makes a model better.

If you train a deeper model than 18L-18L, please set `--clip-norm` to 0.1.

### Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 4 --lenpen 0.6 --remove-bpe | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > generated.result
```

* We used ```--lenpen 0.6``` for newstest2014, and ```--lenpen 1.0``` for otherwise.


### Compute SacreBLEU score

Detokenize the generated result.

```bash
cat generated.result | $mosesscripts/tokenizer/detokenizer.perl -l de > generated.result.detok
```

* mosesscripts is the PATH to mosesdecoder/scripts

Compute SacreBLEU.

```bash
cat generated.result.detok | sacrebleu -t wmt14/full -l en-de
```

## Acknowledgements
This repository is based on [our previous project](https://github.com/takase/rethink_perturbations), whose large portion is borrowed from [fairseq](https://github.com/pytorch/fairseq).
