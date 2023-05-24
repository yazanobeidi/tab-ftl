# Tabular Federated Transfer Learning


## Setup

Clone this repo and `cd` into `env`.

Create a virtual env with Python 3.11 and enter it:

`python3.11 -m venv ~/.env/env`

`source ~/.env/env/bin/activate`

Install Python requirements

`pip install -r requirements.txt`

Add virtualenv to jupyter (optional):

`ipython kernel install --name "env" --user`

Launch JupyterLab

`jupyter lab`

### Running Experiments

### Covertype

[Download the UCI covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype) and place into `tab-ftl/cover/data`. Note this has already been done.

Two of the columns have been one-hot-encoded. Transform these back into ordinal encodings. This reduces the number of features from 54 to 11.

`python datasets/cover/convert_cover54_to_cover11.py --path-to-dataset datasets/cover/covtype.data --output-folder datasets/cover/ --verbose True`

Split the original dataset into 4 geographically-distributed datasets. The default `--partition-key` is `wilderness_area` which replicates the paper.

Note, we do this to simulate a Federated Learning scenario in a plausible manner. In this case we presume each jurisdiction maintains their own records and regulation prohibts data sharing.

If you used the `--output-folder` argument as above, or the default one, then you can just run the following. Otherwise specify the locations of the train, test and validation sets.

`python datasets/cover/partition.py`

Start an experiment run. 

*Note, we include `-W ignore::UserWarning` to suppress this torch warning `UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/NestedTensorImpl.cpp:180`*

For example, with full-federation:

*Note that since client target spaces are inherently mismatched, we must force alignment. This is necessary for aggregation which requires consistent layer weight shapes.*

`python -W ignore::UserWarning run.py  --federate --no-personalize --force-consistent-target-space`

With both partially personalization and federation:

*Note that we could also specify --force-consistent-target-space but this is not necessary with partial personalization.*

`python -W ignore::UserWarning run.py --federate --personalize`

No federation i.e. models train separately, or, "orphaned":

*Note that --personalize is unused when not Federating*

`python -W ignore::UserWarning run.py --no-federate`

To run all 3 experiment runs at once, just omit these options: `--federated`, `--personalize` and `--force-consistent-target-space`.

`python -W ignore::UserWarning run.py`

There are additional command line options that you can specify:

`python -W ignore::UserWarning run.py --seed 0 --device cpu --epochs 105 --batches 50 --ee-dim 10 --log-graph-and-embeddings --description example_of_a_non_default_run`

To run an experiment with a specified range of random seeds, try the following. Note this was used in the paper.

`python -W ignore::UserWarning run.py --device cpu --epochs 105 --model model --batches 10 --seed-range 8:28`

Results will be printed to console, and to log file. For `cover`, this will be available in `tab-ftl/cover/logs`.

Some experiments have a `--centralized` implementation which will run a single model over the full dataset to serve as a comparison. For example:

`run.py --device cpu --epochs 105 --model model --batches 10 --experiment cover --centralized --seed-range 8:28`

## Tensorboard

To launch tensorboard, run

`tensorboard --logdir=cover/log`

It might be helpful to filter runs with a regex.

For example, to view run 1 accuracy for all experiments:

`^(1.*)/.*accuracy`

To view accuracy metrics for all experiments for runs 56, 45, 55, 51:

`^(56.*)/.*accuracy|^(45.*)/.*accuracy|^(55.*)/.*accuracy|^(51.*)/.*accuracy`

To view personalized auroc metrics for runs 75 and 60 only:

`^(75.*).*per.*/.*auroc|^(60.*).*per.*/.*auroc`

It may also be helpful to add this regex to tensorboard colouring:

`(personalized.*auroc.*Val|personalized.*auroc.*Train|orphan.*auroc.*Val|orphan.*auroc.*Train|federatedsame.*auroc.*Val|federatedsame.*auroc.*Train)`