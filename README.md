# Multi-Hop Knowledge Graph Reasoning with Reward Shaping

Based off the work by [salesforce](https://github.com/salesforce/MultiHopKG) but heavily edited.


# Run

## Poetry (Optional)

Not necessary but you might find it convenient to install poetry:

```sh
sudo apt install poetry
```

Then install all packages here

```sh
poetry install --no-root
```

Then enter the environment

```sh
poetry shell
```

## Data

Make sure you've got your hands on `data-release.tgz` and, while in the repo root, decompress it with:

```sh
tar -xvf data-release.tgz
```

## Actually Running it

```sh
python mlm_training.py
```

## Train and Evaluate KGE Model
If you want to **train** you can find a best config in `multihopkg/exogenous/sun_best_config.sh`, just copy the one you need from there and use it in the command line.

Model will be saved to `models/`

To **evaluate** the model you can use either `kge_evaluation.py` or `kge_train.py`. If you choose the 2nd, you should use this command

`CUDA_VISIBLE_DEVICES=[DEVICE NUMBER] python -u kge_train.py --do_test --cuda -init models/[MODEL NAME] --save_path [PATH TO SAVE RESULTS]`
