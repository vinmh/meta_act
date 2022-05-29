import tempfile
from pathlib import Path

import pandas as pd
from skmultiflow.data.data_stream import DataStream

from meta_act.metalearn import MetaLearner
from meta_act.runner import (
    FixedZRunner, ResultsWriter, join_partial_files, MetaLearnerManager,
    ExperimentRunner, ZMTLRunner, SupervisedRunner
)

METADB_PATH = "./test/data/elecNormNew.csv"


class MockLearner:
    def __init__(self):
        self.samples = 0

    def fit(self, X, y, target_values):
        self.samples += X.shape[0]

    def partial_fit(self, X, y, classes):
        self.fit(X, y, classes)

    def prequential_eval(self, X, y, target_values):
        self.fit(X, y, target_values)

    def predict_proba(self, X):
        return [0.5, 0.5]

    def score(self, X, y):
        return y

    def predict(self, X):
        return [1 for _ in range(X.shape[0])]


class MockMetaLearner(MetaLearner):
    def __init__(self):
        super().__init__(MockLearner)


class MockResultsHandler(ResultsWriter):
    def init_files(self):
        pass

    def _write_data(
            self, handler, sample_n, dataset_name, z_val, accuracy, queries
    ):
        print(
            f"{handler},{sample_n},{dataset_name},{z_val},{accuracy},{queries}"
        )


class MockDriftDetector():
    def __init__(self):
        self.samples = 0

    def add_element(self, X):
        self.samples += 1

    def detected_change(self):
        return self.samples % 2 == 0


def test_runners_init():
    result_handler = MockResultsHandler("", True, True)

    zmtl = ZMTLRunner(result_handler, [], {}, [], MockMetaLearner())
    fixz = FixedZRunner(result_handler, [0.1, 0.2, 0.3])
    supervised = SupervisedRunner(result_handler)

    zmtl.init()
    fixz.init()
    supervised.init()

    assert zmtl.learner is not None
    assert zmtl.drift_model is not None
    assert len(fixz.classifier_models) == 3
    assert sorted(
        [x.z_val for x in fixz.classifier_models]
    ) == [0.1, 0.2, 0.3]
    assert supervised.model is not None


def test_runners_pretrain(datadir):
    result_handler = MockResultsHandler("", True, True)

    df = pd.read_csv(datadir / "elecNormNew.csv")
    stream = DataStream(df)

    X, y = stream.next_sample(10)

    zmtl = ZMTLRunner(result_handler, [], {}, [], MockMetaLearner())
    fixz = FixedZRunner(result_handler, [0.1, 0.2, 0.3])
    supervised = SupervisedRunner(result_handler, MockLearner)

    zmtl.init()
    fixz.init()
    supervised.init()

    zmtl.pretrain(X, y, stream.target_values)
    fixz.pretrain(X, y, stream.target_values)
    supervised.pretrain(X, y, stream.target_values)

    assert zmtl.learner.hits + zmtl.learner.miss == 10
    for learner in fixz.classifier_models:
        assert learner.hits + learner.miss == 10
    assert supervised.model.hits + supervised.model.miss == 10


def test_runners_next_data(datadir):
    result_handler = MockResultsHandler("", True, True)

    df = pd.read_csv(datadir / "elecNormNew.csv")
    stream = DataStream(df)

    X, y = stream.next_sample()

    zmtl = ZMTLRunner(result_handler, [], {}, [], MockMetaLearner())
    fixz = FixedZRunner(result_handler, [0.1, 0.2, 0.3])
    supervised = SupervisedRunner(result_handler, MockLearner)

    zmtl.init()
    fixz.init()
    supervised.init()

    zmtl.next_data(1, "abc", X, y, stream.target_values, True)
    fixz.next_data(1, "abc", X, y, stream.target_values, True)
    supervised.next_data(1, "abc", X, y, stream.target_values, True)

    assert zmtl.learner.samples_seen == 1
    for learner in fixz.classifier_models:
        assert learner.samples_seen == 1
    assert supervised.model.samples_seen == 1


def test_zmtl_runner_drift(datadir):
    result_handler = MockResultsHandler("", True, True)

    df = pd.read_csv(datadir / "elecNormNew.csv")
    stream = DataStream(df)

    X1, y1 = stream.next_sample()
    X2, y2 = stream.next_sample()

    meta_learner = MockMetaLearner()
    meta_learner.trained = True

    zmtl = ZMTLRunner(
        result_handler, ["nr_attr"], None, None, meta_learner,
        drift_detector=MockDriftDetector
    )

    zmtl.init()

    zmtl.next_data(1, "abc", X1, y1, stream.target_values, True)
    zmtl.next_data(2, "abc", X2, y2, stream.target_values, True)

    assert zmtl.learner.z_val == 1


def test_results_writer():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path1 = (Path(tmpdir) / "results1.csv").as_posix()
        base_path2 = (Path(tmpdir) / "results2.csv").as_posix()

        writer1 = ResultsWriter(
            base_path1, True, True
        )
        writer2 = ResultsWriter(
            base_path2, False, True
        )
        writer1.init_files()
        writer2.init_files()

        writer1.write_headers()
        writer2.write_headers()

        writer1.write_zmtl(10, "hello", 0.1, 0.8, 2)
        writer1.write_fixed_z(10, "bye", 0.1, 0.8, 2)
        writer1.write_supervised(10, "okay", 0.1)

        writer2.write_zmtl(11, "hello", 0.1, 0.8, 2)
        writer2.write_fixed_z(11, "bye", 0.1, 0.8, 2)
        writer2.write_supervised(11, "okay", 0.1)

        writer1.close_all()
        writer2.close_all()

        files_found1 = list(Path(tmpdir).glob("results1.csv.*"))
        files_found2 = list(Path(tmpdir).glob("results2.csv.*"))

        assert len(files_found1) == 3
        assert len(files_found2) == 2


def test_join_partial_files(datadir):
    file_list = list((datadir / "join_test").glob("*.*"))
    join_partial_files(file_list)

    f1 = datadir / "join_test/result1_zmtl.csv"
    f2 = datadir / "join_test/result1_fixed.csv"

    assert f1.exists()
    assert f2.exists()

    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    print(df1)
    print(df2)

    assert df1.shape == (6, 3)
    assert df2.shape == (9, 3)


def test_train_metamodel(datadir):
    metadb = pd.read_csv(datadir / "metadb_test.csv")

    manager = MetaLearnerManager(train_metadb_metadata_size=10)
    learner, results = manager.train_metamodel(metadb)

    assert learner is not None
    assert results["R^2-Train"] is not None


def test_test_metamodel(datadir):
    metadb = pd.read_csv(datadir / "metadb_test.csv")

    manager = MetaLearnerManager(train_metadb_metadata_size=10)
    results = manager.test_metamodel(metadb, 5)

    print(results)

    assert results is not None
    assert results.shape == (5,)


def test_partition_datasets_per_thread():
    runner = ExperimentRunner(
        experiments_threads=4
    )

    dataset_list = [str(x) for x in range(10)]

    datasets_per_thread = runner._partition_datasets_per_thread(
        dataset_list
    )
    print([len(x) for x in datasets_per_thread])

    assert len(datasets_per_thread[0]) == 4
    assert len(datasets_per_thread[1]) == 2
    assert len(datasets_per_thread[2]) == 2
    assert len(datasets_per_thread[3]) == 2


def test_experiment_thread(datadir):
    runner = ExperimentRunner(
        classifier=MockLearner,
        experiments_mfe_features=["nr_attr"],
        drift_detector=MockDriftDetector,
        experiments_results_save_delay=1,
        experiments_grace_period=2
    )

    metalearner = MockMetaLearner()
    metalearner.trained = True

    datafiles = list((datadir / "exp_datasets").glob("*.csv"))
    outpath = datadir / "results.csv"

    runner._experiment_thread(metalearner, datafiles, outpath.as_posix())

    print(list(datadir.glob("*.*")))

    r1 = datadir / "results.csv.zmtl"
    r2 = datadir / "results.csv.fixz"
    r3 = datadir / "results.csv.trad"

    assert r1.exists()
    assert r2.exists()
    assert r3.exists()

    results1 = pd.read_csv(r1)
    results2 = pd.read_csv(r2)
    results3 = pd.read_csv(r3)

    assert (results1.groupby(by=["dataset_name"]).count().shape[0]
            == len(datafiles))
    assert (results2.groupby(by=["dataset_name"]).count().shape[0]
            == len(datafiles))
    assert (results3.groupby(by=["dataset_name"]).count().shape[0]
            == len(datafiles))


def test_run_experiments(datadir):
    datasets_path = datadir / "exp_datasets"
    outpath = datadir / "results.csv"
    runner = ExperimentRunner(
        benchmark_dataset_dir=datasets_path,
        experiments_mfe_features=["nr_attr"],
        classifier=MockLearner,
        drift_detector=MockDriftDetector,
        experiments_results_save_delay=1,
        experiments_threads=2,
        experiments_grace_period=2
    )

    metalearner = MockMetaLearner()
    metalearner.trained = True

    runner.run_experiments(metalearner, outpath)

    r1 = datadir / "results_zmtl.csv"
    r2 = datadir / "results_fixz.csv"
    r3 = datadir / "results_trad.csv"

    assert datadir / "results_zmtl.csv"
