import numpy as np

class UniformDist:
    def __init__(self, min_val, max_val, is_int=False):
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int

    def sample(self, n_samples):
        samples = np.round(np.random.uniform(self.min_val, self.max_val, n_samples), 4)
        if self.is_int:
            samples = np.round(samples).astype(int)
            samples = np.clip(samples, self.min_val, self.max_val)
            
        return np.unique(samples)[:n_samples]  # Ensure unique samples

class NormalDist:
    def __init__(self, mean, std, min_val=None, max_val=None, is_int=False):
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int

    def sample(self, n_samples):
        samples = np.random.normal(self.mean, self.std, n_samples)
        if self.min_val is not None or self.max_val is not None:
            samples = np.round(np.clip(samples, self.min_val or -np.inf, self.max_val or np.inf), 4)
        if self.is_int:
            samples = np.round(samples).astype(int)
            samples = np.clip(samples, self.min_val or -np.inf, self.max_val or np.inf)
        return np.unique(samples)[:n_samples]

class CategoricalDist:
    def __init__(self, options):
        self.options = options

    def sample(self, n_samples):
        if n_samples <= len(self.options):
            return np.random.choice(self.options, n_samples, replace=False)
        else:
            return np.random.choice(self.options, n_samples, replace=True)

    
def sample_param_grid(param_grid, n_samples):

    sampled_grid = {}
    for param_name, dist in param_grid.items():
        sampled_grid[param_name] = dist.sample(n_samples)
    return list(sampled_grid.keys()), list(sampled_grid.values())