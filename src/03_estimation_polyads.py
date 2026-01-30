import numpy as np
import pandas as pd
from polyads.data import generate_data
from polyads.model import PolyadEstimator

# Generate synthetic three-way gravity data
beta_true = np.array([1.0, -0.5])
df, X = generate_data(
    seed=1,
    n_ds=(100, 100, 100),        # Dimensions: n1 × n2 × n3
    c=-5,                         # Baseline intensity
    shape=np.inf,                 # Poisson model
    beta=beta_true,               
    groups=[[0,1], [0,2], [1,2]] # Three-way fixed effects
)

# Fit the model
columns = df.columns.tolist()
estimator = PolyadEstimator(use_tqdm=True)
estimator.fit(
    df=df, 
    indices=columns[:-1],         # Index columns
    values=columns[-1],           # Count column
    beta_init=np.zeros(2),
    X=X
)

# Display results
estimator.summary()