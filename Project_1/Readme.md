This is our Project 1 source code.

* `core` contains the training loop and model evaluation
* `run_experiments` contains infrastructure for easily running an experiment for multiple rounds with newly generated
  data and initialized weights
* `models` contains some utility Modules for building models
* `util` contains various utilities related to device seletion, data generation, plotting, ...

* `test` is the main file to run, it reproduces the `Shared a=10` experiment from the report but only runs for 3 rounds,
  to make it a bit faster to run on the VM.
* `experiment_report` reproduces all the experiments we talk about in the report
* `experiment_explorer` contains different experiments used during the project, some of which we later dropped.
