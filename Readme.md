# Shallow Water Semi-Analytical Framework

A Python framework for shallow water semi-analytical models and parameter inference for coastal remote sensing.

## Features

- Multiple backend support (JAX, NumPy, TensorFlow, Numba)
- Flexible parameter management
- Advanced statistical inference capabilities
- Comprehensive optimization framework
- Extensive visualization tools
- Support for various sensor models 

## Installation

```bash
git clone [repository-url]
cd shallow-water-framework
pip install -r requirements.txt
```

## Quick Start

### 1. Simulating Reflectance Spectra

```python
from shallow_sa.model.radiometry.wraper import ReflectanceSimulator

# Initialize simulator
model = ReflectanceSimulator(
    config_path='./config_model.yml',
    prop_type='rrsp',
    sensor='hyspex_muosli',
    backend='TENSORFLOW'
)

# Define parameters
params = {
    'z': 8,            # Depth in meters
    'alpha_m_1': .2,   # Mixing ratio of mineral 1 and vegetation 0
    'chl': -7,         # Chlorophyll
    'cdom': -7,        # Colored dissolved organic matter
    'tsm':-23          # Total suspended matter
}

# Generate and plot reflectance
reflectance = model.simulate(params)
model.plot(reflectance, "Sample Spectral Response")
```

### 2. Parameter Profiling and Inference

```python
from shallow_sa.wrappers.run_inference import ShallowSAInference

# Initialize inference
sa_infer = ShallowSAInference(config_path='config_model.yml')

# Generate sample data
params = {
    "z": 2,
    "cdom": -4,
    "chl": -7,
    "tsm": -6,
    "alpha_m_1": .2
}
sample = sa_infer.generate_sample(params=params, n_samples=1)[0]

# Profile parameter
profile_data = sa_infer.profile_parameter(
    "z",
    sample,
    init_param=params,  # Initialize from true value for simulation
    num_points=25,
    values='diffuse'
)

# Access results
print(profile_data.confidence_intervals)
print(profile_data.standard_errors)

# Visualize results
sa_infer.plot_profile(profile_data)
```

## Configuration

The framework uses YAML configuration files to manage various settings. Create a `config_model.yml` file:

```yaml
backend: JAX  # Options: NUMPY, TENSORFLOW, JAX, NUMBA

design:
  sensor: sentinel_2A  # Options: sentinel_2B, sentinel_2A, dirac, hyspex_muosli
  min_wvl: 400
  max_wvl: 800
  wvl_step: 1

modeling_args:
    aphy_model_option: "log_bricaud"
    acdom_model_option: "log_bricaud"
    anap_model_option: "log_bricaud"
    bb_chl_model_option: "None"
    bb_cdom_model_option: "None"
    bb_tsm_model_option: "log_bricaud"
    bottom_model_option: "linear"  # or "sigmoid"

prop_type: "rrsp"

optimizer:
    method: trust-constr
    bounded_opt: True
    initialization: "Theta_True"
    prop: "rrsp"

parameterization:
    activation_flags: {
        'chl': 1,
        'cdom': 1,
        'tsm': 1,
        'z': 1,
        'alpha_m_1': 1
    }

profiler:
    deviation: 6

simulation:
    parameterization: TWBB
    psi_key: z
    n_mc: 2
    std_noise_log: -3.5
    verbose: 0

paths:
    param_file_path: "src/config/param_space.json"
    results_directory: "./results"
```

## Key Components

### Reflectance Simulator
- Simulates reflectance spectra based on input parameters
- Supports multiple sensor models
- Configurable radiative transfer models

### Parameter Inference
- Statistical inference for parameter estimation
- Multiple optimization methods
- Confidence interval calculation
- Parameter profiling capabilities

### Visualization
- Spectral response plots
- Parameter profile visualization
- Confidence interval plots
- Diagnostic visualizations

## Advanced Configuration

### Parameter Activation Flags
- All parameters unknown:
```yaml
activation_flags: {'chl': 1, 'cdom': 1, 'tsm': 1, 'z': 1, 'alpha_m_1': 1}
```

- Known water column IOPs:
```yaml
activation_flags: {'chl': 0, 'cdom': 0, 'tsm': 0, 'z': 1, 'alpha_m_1': 1}
```

- Known water column depth and bottom reflectance:
```yaml
activation_flags: {'chl': 1, 'cdom': 1, 'tsm': 1, 'z': 0, 'alpha_m_1': 0}
```

### Backend Selection
Choose the computational backend based on your needs:
- JAX: For automatic differentiation and acceleration
- NumPy: For standard numerical operations
- TensorFlow: For deep learning integration
- Numba: For JIT compilation

## Parameter Descriptions

- `z`: Depth in meters
- `alpha_m_1`: Mixing ratio of mineral 1 and vegetation 0
- `chl`: Chlorophyll concentration (log scale)
- `cdom`: Colored dissolved organic matter (log scale)
- `tsm`: Total suspended matter (log scale)

## Best Practices

1. Always validate configuration before running simulations
2. Choose appropriate backend for your use case
3. Consider computational resources when setting sample sizes
4. Use parameter profiling for uncertainty estimation
5. Validate results with known test cases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify License]

## Citation

[Specify Citation Information]