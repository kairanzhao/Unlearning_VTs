## Original model training
```
python main_train.py --dataset {dataset name} --arch {model architecture} --epochs {epochs} --lr {learning_rate} --batch_size {batch_size}
```

## Unlearning performance

### Retrain

```
python main_forget.py --unlearn retrain --num_indexes_to_replace 3000 --unlearn_epochs 30 --unlearn_lr 0.1 --mem mix --batch_size 256
```

### RUM with memorization proxy (e.g. NegGrad+, Fine-Tune, SalUn, etc.)

```
python main_rum_proxy.py --unlearn ${unlearn method} --mem mix --num_indexes_to_replace 3000 --dataset {dataset name} --arch {model architecture} --epochs {epochs for training the original model} --lr 0.1 --batch_size 256
```


### Evaluation using ToW / ToW-MIA

```
python analysis.py --dataset {dataset name} --arch {model architecture} --no_aug --unlearn ${unlearn method} --mem_proxy {memorization proxy} --mem {memorization group} --num_indexes_to_replace 3000
```
```
python analysis_mia.py --dataset {dataset name} --arch {model architecture} --no_aug --unlearn ${unlearn method} --mem_proxy {memorization proxy} --mem {memorization group} --num_indexes_to_replace 3000
```

## References


[RUM] (https://github.com/kairanzhao/RUM)

[SalUn] (https://github.com/OPTML-Group/Unlearn-Saliency)

[SCRUB] (https://github.com/meghdadk/SCRUB)

[Heldout Influence Estimation] (https://github.com/google-research/heldout-influence-estimation)

[Data Metrics] (https://github.com/meghdadk/data-metrics)

