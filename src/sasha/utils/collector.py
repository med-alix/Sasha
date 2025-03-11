import json
import os
import numpy as np
from tqdm.notebook import tqdm


class DataCollector:
    def __init__(self, keys, parameterization, bands):
        self.data = {"n_samples": 0}
        self.data["parameterization_desc"] = parameterization
        self.data["bands"] = self.np_array_to_list(bands)
        self.data["parameter_keys"] = keys  # New 1;

    def add(self, data_dict):
        for key, value in data_dict.items():
            # Initialize the key with an empty list if it does not exist
            if key not in self.data:
                self.data[key] = []
            # Append the value to the list associated with the key
            self.data[key].append(self.np_array_to_list(value))

    def np_array_to_list(self, data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self.np_array_to_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.np_array_to_list(element) for element in data]
        else:
            return data

    def complete_data(self):
        self.data["n_samples"] = len(self.data["lp"])
        for key in self.data["parameter_keys"]:
            key_mle = "mle_" + key
            if key_mle in self.data:
                values = self.data[key_mle]
                if len(values) > 0:
                    median = np.median(values)
                    mean = np.mean(values)
                    sup95 = np.percentile(values, 95)
                    inf95 = np.percentile(values, 5)
                    self.data[key + "_median"] = median
                    self.data[key + "_mean"] = mean
                    self.data[key + "_sup95"] = sup95
                    self.data[key + "_inf95"] = inf95

    def save_to_json(self, filepath):
        self.complete_data()
        with open(filepath, "w") as file:
            json.dump(self.data, file, indent=4)


class AggregatedResultsCollector:
    def __init__(self, profiler, file_path):
        self.profiler = profiler
        self.file_path = file_path
        self.aggregated_data = self._load_or_initialize_aggregated_data()

    def _load_or_initialize_aggregated_data(self):
        """Attempt to load existing aggregated data or initialize a new structure."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as file:
                    data = json.load(file)
                # Perform validation or initialization if file is empty/not correctly formatted
                if not isinstance(data, dict):
                    return self._initialize_new_aggregated_structure()
                return data
            except json.JSONDecodeError:
                # File exists but failed to decode; start fresh
                return self._initialize_new_aggregated_structure()
        else:
            return self._initialize_new_aggregated_structure()

    def _initialize_new_aggregated_structure(self):
        """Define and return a new, empty aggregated data structure."""
        # Initialize based on your expected result structure
        return {
            "mle_initial": {"alpha_m_1": [], "z": []},
            # Extend with other fields as needed
        }

    def _aggregate_data(self, new_results):
        """Aggregate new data into the existing aggregated data structure, handling specific requirements."""
        for key, value in new_results.items():
            if key in ["model", "reflectance_model", "wvl", "bands"]:
                self.aggregated_data[key] = value.tolist()
            elif key in ["true_value"]:
                self.aggregated_data[key] = value
            elif isinstance(value, dict):
                # Handle dictionaries (like 'mle_initial', 'nuisance_mle')
                if key not in self.aggregated_data:
                    self.aggregated_data[key] = {}
                for sub_key, sub_value in value.items():
                    if sub_key not in self.aggregated_data[key]:
                        self.aggregated_data[key][sub_key] = []
                    # For arrays, ensure they're converted to lists before appending
                    if isinstance(sub_value, np.ndarray):
                        sub_value = sub_value.tolist()
                    self.aggregated_data[key][sub_key].append(sub_value)
            elif isinstance(value, str):
                if key in ["interest_key", "parameterization_desc"]:
                    # Avoid duplication for 'interest_key'
                    self.aggregated_data[key] = value
            elif isinstance(value, list):
                # For lists (like 'interest_values'), extend if not 'interest_key'
                if key not in self.aggregated_data:
                    self.aggregated_data[key] = []
                else:
                    # Convert numpy arrays to lists if necessary
                    if any(isinstance(x, np.ndarray) for x in value):
                        value = [
                            x.tolist() if isinstance(x, np.ndarray) else x
                            for x in value
                        ]
                    self.aggregated_data[key].extend(value)
            else:
                # For scalar values (like 'interest_key'), directly append
                if key not in self.aggregated_data:
                    self.aggregated_data[key] = []
                self.aggregated_data[key].append(value)
        # Special handling for 'ci' to ensure it's aggregated as lists of two elements per sample
        if "ci" in new_results:
            for ci_key, ci_value in new_results["ci"].items():
                if ci_key not in self.aggregated_data["ci"]:
                    self.aggregated_data["ci"][ci_key] = []
                # Convert numpy arrays to lists if necessary and ensure two elements per sample
                if isinstance(ci_value, np.ndarray):
                    ci_value = ci_value.tolist()
                self.aggregated_data["ci"][ci_key].append(ci_value)

    def collect_and_aggregate(self, sample, num_points=15, width=1, **kwargs):
        """Profile and aggregate results from a single sample."""
        profiled_data = self.profiler.profile_uniform(
            "z", sample, num_points=num_points, width=width
        )
        analysis_results = self.profiler.perform_post_profiling_analysis(
            profiled_data, sample
        )
        self._aggregate_data(analysis_results)

    def _convert_numpy_to_list(self, obj):
        """Recursively convert numpy arrays in the object to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(v) for v in obj]
        return obj

    def complete_data(self):
        sup95, inf95, mean = {}, {}, {}
        for key, val in self.aggregated_data["mle_final"].items():
            sup95[key] = np.nanquantile(val, 0.975)
            inf95[key] = np.nanquantile(val, 0.025)
            mean[key] = np.nanmean(val)
        self.aggregated_data["sup95"] = sup95
        self.aggregated_data["inf95"] = inf95
        self.aggregated_data["mean"] = mean

    def save_aggregated_data(self):
        """Save the aggregated data back to the JSON file, ensuring all data is serializable."""

        def default(obj):
            """JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            try:
                return np.array(obj).tolist()
            except ValueError:
                return str(
                    obj
                )  # As a fallback, convert to string if unable to convert to list

        self.complete_data()
        serializable_data = self._convert_numpy_to_list(self.aggregated_data)
        directory = os.path.dirname(self.file_path)
        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.file_path, "w") as file:
            json.dump(serializable_data, file, indent=4, default=default)

    def run_collection_loop(
        self, mvg_inference, theta_true, iterations, num_points=15, width=1, **kwargs
    ):
        """Collect and aggregate data across multiple samples."""
        for _ in tqdm(range(iterations)):
            sample = mvg_inference.generate_samples(1, **theta_true).reshape(1, -1)
            self.collect_and_aggregate(sample, num_points=num_points, width=width)
        self._aggregate_data(kwargs)
        self.aggregated_data["n_samples"] = iterations
        self.save_aggregated_data()

    def run_collection_range(
        self, mvg_inference, theta_true, interval, num_points=15, width=1, **kwargs
    ):
        """Collect and aggregate data across multiple samples."""
        for _ in range(interval[0], interval[1]):
            sample = mvg_inference.generate_samples(1, **theta_true).reshape(1, -1)
            self.collect_and_aggregate(sample, num_points=num_points, width=width)
        file_path_old = self.file_path
        self.file_path = (
            f"{file_path_old.strip('.json')}_{interval[0]}_{interval[1]}.json"
        )
        self._aggregate_data(kwargs)
        self.aggregated_data["n_samples"] = interval[1] - interval[0]
        self.save_aggregated_data()
