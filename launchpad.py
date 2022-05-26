import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from swing_data_instance import *


class LaunchpadClassifier:
    def is_swing(self, samples: list[WristSample]) -> bool:
        pass


class RocketPuttingRidge(LaunchpadClassifier):
    def post_process(self, pd, classes, crop) -> pd.DataFrame:
        return skd_post_process(pd, classes,
                                classes_to_remove=[5, 6, 7],
                                class_remap={1: 0, 2: 0, 3: 0, 4: 0},
                                crop_series_rows=crop,
                                dimensions_to_remove=self.dimensions_to_remove,
                                synthesize_dimensions=self.synthesize_dimensions)

    def __init__(self,
                 split_path: str,
                 crop: slice | None = None,
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = []):
        self.crop = crop
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions

        train_data, test_data = sdi_load_split(split_path)
        train_pd, train_classes = sdiList2sktimeData(train_data)
        test_pd, test_classes = sdiList2sktimeData(test_data)
        train_pd = self.post_process(train_pd, train_classes, crop)
        test_pd = self.post_process(test_pd, test_classes, crop)
        self.train_data_size = len(train_pd.iat[0, 0])

        train_data.clear()
        test_data.clear()

        print("Initializing RocketPuttingRidge...")
        self.pipeline = make_pipeline(
            MiniRocketMultivariate(), StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            verbose=True
        )

        self.pipeline.fit(train_pd, train_classes)
        self_score = self.pipeline.score(test_pd, test_classes)
        print(f"Initialized with score {self_score}")

    def is_swing(self, samples: list[WristSample]) -> bool:
        data = wristSample2sktimeData(samples)
        data = self.post_process(data, [8], None)
        if len(data.iat[0, 0]) != self.train_data_size:
            # print(f"Should test with the same amount of samples as was training data length ({len(samples)} vs {self.train_data_size})")
            return False
        predictions = self.pipeline.predict(data)
        return predictions[0] == 8


class RocketFullSwingRidge(LaunchpadClassifier):
    def post_process(self, pd, classes, crop) -> pd.DataFrame:
        return skd_post_process(pd, classes,
                                classes_to_remove=[1, 2, 3, 5, 6, 7, 8],
                                #class_remap={5: 0, 6: 0, 7: 0, 8: 0},
                                crop_series_rows=crop,
                                dimensions_to_remove=self.dimensions_to_remove,
                                synthesize_dimensions=self.synthesize_dimensions)

    def __init__(self,
                 split_path: str,
                 crop: slice | None = None,
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = []):
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions

        train_data, test_data = sdi_load_split(split_path)
        train_pd, train_classes = sdiList2sktimeData(train_data)
        test_pd, test_classes = sdiList2sktimeData(test_data)
        train_pd = self.post_process(train_pd, train_classes, crop)
        test_pd = self.post_process(test_pd, test_classes, crop)
        self.train_data_size = len(train_pd.iat[0, 0])

        train_data.clear()
        test_data.clear()

        print("Initializing RocketFullSwingRidge...")
        self.pipeline = make_pipeline(
            MiniRocketMultivariate(), StandardScaler(with_mean=False), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
            verbose=True
        )

        self.pipeline.fit(train_pd, train_classes)
        self_score = self.pipeline.score(test_pd, test_classes)
        print(f"Initialized with score {self_score}")

    def is_swing(self, samples: list[WristSample]) -> bool:
        data = wristSample2sktimeData(samples)
        data = self.post_process(data, [4], None)
        if len(data.iat[0, 0]) != self.train_data_size:
            # print(f"Should test with the same amount of samples as was training data length ({len(samples)} vs {self.train_data_size})")
            return False
        predictions = self.pipeline.predict(data)
        return predictions[0] == 4


class RocketPuttingIsolation(LaunchpadClassifier):
    def post_process(self, pd, classes, crop) -> pd.DataFrame:
        return skd_post_process(pd, classes,
                                classes_to_remove=[5, 6, 7],
                                class_remap={1: 0, 2: 0, 3: 0, 4: 0},
                                crop_series_rows=crop,
                                dimensions_to_remove=self.dimensions_to_remove,
                                synthesize_dimensions=self.synthesize_dimensions)

    def __init__(self,
                 split_path: str,
                 crop: slice | None = None,
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = []):
        self.crop = crop
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions

        train_data, test_data = sdi_load_split(split_path)
        train_pd, train_classes = sdiList2sktimeData(train_data)
        train_pd = self.post_process(train_pd, train_classes, crop)
        self.train_data_size = len(train_pd.iat[0, 0])

        train_data.clear()

        print("Initializing RocketPuttingIsolation...")
        self.pipeline = make_pipeline(
            MiniRocketMultivariate(), StandardScaler(with_mean=False), IsolationForest(),
            verbose=True
        )

        self.pipeline.fit(train_pd)

    def is_swing(self, samples: list[WristSample]) -> bool:
        data = wristSample2sktimeData(samples)
        data = self.post_process(data, [8], None)
        if len(data.iat[0, 0]) != self.train_data_size:
            # print(f"Should test with the same amount of samples as was training data length ({len(samples)} vs {self.train_data_size})")
            return False
        predictions = self.pipeline.predict(data)
        return predictions[0] == 1


class RocketFullSwingIsolation(LaunchpadClassifier):
    def post_process(self, pd, classes, crop) -> pd.DataFrame:
        return skd_post_process(pd, classes,
                                classes_to_remove=[0, 1, 2, 3, 5, 6, 7, 8],
                                crop_series_rows=crop,
                                dimensions_to_remove=self.dimensions_to_remove,
                                synthesize_dimensions=self.synthesize_dimensions)

    def __init__(self,
                 split_path: str,
                 crop: slice | None = None,
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = []):
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions

        train_data, test_data = sdi_load_split(split_path)
        train_pd, train_classes = sdiList2sktimeData(train_data)
        train_pd = self.post_process(train_pd, train_classes, crop)
        self.train_data_size = len(train_pd.iat[0, 0])

        train_data.clear()

        print("Initializing RocketFullSwingIsolation...")
        self.pipeline = make_pipeline(
            MiniRocketMultivariate(), StandardScaler(with_mean=False), IsolationForest(),
            verbose=True
        )

        self.pipeline.fit(train_pd)

    def is_swing(self, samples: list[WristSample]) -> bool:
        data = wristSample2sktimeData(samples)
        data = self.post_process(data, [4], None)
        if len(data.iat[0, 0]) != self.train_data_size:
            # print(f"Should test with the same amount of samples as was training data length ({len(samples)} vs {self.train_data_size})")
            return False
        predictions = self.pipeline.predict(data)
        return predictions[0] == 1
