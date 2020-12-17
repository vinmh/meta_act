# MetaRecommender for Entropy Threshold based Active Learning

# (Text below if for future reference, nothing works yet)

## Introduction

This library aims to provide utilities for use in meta-recommending entropy
thresholds (z values) for Active Learning.

Right now only stream-based active learning is supported.

## To-Do

- Metadatabase generator
- Metalearner
- Metarecommender exporter

## Usage

This library provides the following utilities:

### Metadatabase generation

This library generates metadatabases to train a metarecommender of
your choice. This works by either using pre-specified datasets or
generating them on the fly, alternating between a different set of
generators.

The generators are taken from
[scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow),
the available ones are:

- HyperplaneGenerator
- LEDGeneratorDrift
- MIXEDGenerator
- RandomRBFGeneratorDrift
- RandomTreeGenerator
- SineGenerator
- STAGGERGenerator

The generator can be invoked like this:
```meta_act.ds_gen.generate_datasets("HyperplaneGenerator", "./datasets", max_samples=100000, **gen_kwargs)```

This will generate one dataset with the _HyperplaneGenerator_
with 10000 samples and save it at `./datasets/`, the _gen_kwargs_
are passed on to the generator constructor.

The actual metadatabase can be generated with the following function:
```
def create_metadb(
    datasets=datasets,
    threads=2,
    delta=0.0001,
    z_vals_n=4,
    z_val_base=2
    hf_kwargs={},
    pre_train_size=300,
    grace_period=300,
    z_errormargin=0.02,
    mfe_feats=["nr_class", "attr_ent", "class_ent", "kurtosis", "skewness"],
    tsfel_domains=["temporal"],
    output_path="./metadb.csv",
    verbose_mode="inline"
)
```
This will generate a metadatabase file called _metadb.csv_ with the
specified datasets generated earlier with the specified MFE metafeatures
and the Tsfel domain metafeatures. It will also work in 2 threads
simultaneously.

The _z_vals_n_ parameter specify how many z values each dataset will be
evaluated against, the z values are generated according to the
max entropy of the dataset (`log(n_classes, base=z_vals_base)`),
for example if _z_vals_n_ = 4 and the max entropy is 1.0, the z values
being evaluated will be 0.2, 0.4, 0.6 and 0.8.

Datasets for metadatabase can also be generated on demand like this:
```
def create_metadb(
    dataset_generators=["HyperplaneGenerator", "LEDGeneratorDrift"],
    dataset_n=500,
    threads=2,
    delta=0.0001,
    z_vals_n=4,
    z_val_base=2
    hf_kwargs={},
    pre_train_size=300,
    grace_period=300,
    z_errormargin=0.02,
    mfe_feats=["nr_class", "attr_ent", "class_ent", "kurtosis", "skewness"],
    tsfel_domains=["temporal"],
    output_path="./metadb.csv",
    verbose_mode="inline"
)
```
This will generate 500 datasets, alternating between the two generators
chosen with randomized generator parameters.
