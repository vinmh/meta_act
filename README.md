# MetaRecommender for Entropy Threshold based Active Learning

![Coverage](./coverage.svg)

## Introduction

This library aims to provide utilities for use in meta-recommending entropy thresholds (z values) for Active Learning.

Right now only stream-based active learning is supported.

## To-Do

- Tests

## Usage

This library provides the following utilities:

### Metadatabase generation

This library generates metadatabases to train a metarecommender of your choice. This works by either using pre-specified
datasets or generating them on the fly, alternating between a different set of generators.

The generators are taken from
[scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow), the available ones are:

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

There is also an alternate generator that, instead of saving dataset files, it returns a python generator function that
returns in memory datasets created from alternating generators with randomized hyperparameters. For example:

```
meta_act.ds_gen.dataset_generator([("HyperplaneGenerator",
                                   {"n_features": [5, 10, 15]}),
                                   ("LEDGeneratorDrift", {"noise_percentage": [0.0, 0.1, 0.5]}],
                                   max_samples=100000)
```

This will generate infinite datasets by alternating between the _HyperplaneGenerator_
and the _LEDGeneratorDrift_ and each time selecting a random value for the hyperparemeters, in the case of the _
HyperplaneGenerator_ the hyperparameter would be _n_features_, and in the case of _LEDGeneratorDrift_ it would be _
noise_percentage_. Do note that as many hyperparameters as desired are possible, in fact it is encouraged to use as many
as possible. For numeric hyperparameters, the possible values can be easily set through ```list(range(x,y,z))```.

The actual metadatabase can be generated with the following function:

```
meta_act.metadb_craft.create_metadb(stream_files,
                                    z_vals_n=5,
                                    z_val_selection_margin=0.02,
                                    window_pre_train_sample_n=300,
                                    window_adwin_delta=0.0001,
                                    stop_conditions={"minority_target": 100},
                                    max_failures=100)
```

There are more arguments that can be passed, check the code for specifications. If _stream_files_ is sent as a list of
strings, it assumes a list of filepaths was sent and it will load them individually, if it detects a generator, it will
take the datasets from it, this is intented to be used with the python generator above.

Stop conditions are conditions that will interrupt the metadatabase creation independently of the amount of stream files
remaining, this needs to be set if the _dataset_generator_ is being used with infinite datasets, otherwise it will enter
an infinite loop until the computer runs out of memory.Possible stop conditions are the following:

- ```max_datasets```: Maximum number of used datasets;
- ```max_samples```: Maximum number of samples in the metadatabase;
- ```minority_target```: Maximum number of samples with the minority z value
- ```majority_target```: Maximum number of samples with the majority z value

Multiple stop conditions may be set as well.

The _z_vals_n_ parameter specify how many z values each dataset will be evaluated against, the z values are generated
according to the max entropy of the dataset (`log(n_classes, base=z_vals_base)`), for example if _z_vals_n_ = 4 and the
max entropy is 1.0, the z values being evaluated will be 0.2, 0.4, 0.6 and 0.8.

If the parameter _output_path_ is not set, the metadatabase is returned in-memory from the function, otherwise it will
save as a csv file in the specified path.

The parameter _max_failures_ must be set to a reasonable large number preferably, it will determine the amount of
datasets that can fail before aborting the metadatabase generation, specially important when using an infinite generator
since in case of errors, stop conditions may never be achieved.

### Online Window Extraction and MetaLearning

Window features may be extracted from a stream window with the function:

```
meta_act.windows.get_window_features(X, mfe_features, tsfel_config, summary_funcs)
```

This will return a single line of features to be used with the metalearner, X is expected to be a numpy array.

With this, its possible to use the MetaLearner class from
`meta_act.metalearn.MetaLearner(learner, *learner_args, **learner_kwargs)`. If the learner parameter is set to a string,
it will assume an already trained model is being attempted to be loaded, `joblib` is required, in this case the
parameters `learner_args` and `learner_kwargs` are simply ignored. Otherwise the learner parameter is treated as a
sklearn algorithm class and will attempt to initialize it with the args and kwargs sent.

`meta_act.metalearn.MetaLearner.fit(X, y, oversample=True, test_data=None)`
can be called to train the model, if oversample is set to `True`, it will attempt to oversample the training dataset
with _SMOTE_. If the parameter `test_data` is set to a tuple, it is assumed it is composed of a two element tuple, the
first being a test X array and the second being a test y array, and the results will include test metrics (R^2 on test
data, MSE and MAE).

After the model is trained, it can be used to predict z values on a number of samples
with `meta_act.metalearn.MetaLearner.predict(X)`, recover metrics from test data
with `meta_act.metalearn.MetaLearner.test(X, y)`
and be saved on a file with `meta_act.metalearn.MetaLearner.save_model(filepath)`. Saving the model two files are
created, a metadata file containing various data about the training environment and the serialized model file to be
loaded like before. The metadata file is only loaded if it is present in the same directory as the serialized model
file.
