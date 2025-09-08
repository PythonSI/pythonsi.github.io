r"""
SI for Lasso Feature Selection
==============================================================================
"""

# Author: Tran Tuan Kiet

from psi import Pipeline
from psi.feature_selection import LassoFeatureSelection
from psi import Data
from psi.test_statistics import FSTestStatistic
import numpy as np

# %%
# Define the pipeline
# -------------------

def pipeline() -> Pipeline:
    x = Data()
    y = Data()
    
    lasso = LassoFeatureSelection(lambda_=10)
    active_set = lasso.run(x, y)
    return Pipeline(inputs=(x, y), output=active_set, test_statistic=FSTestStatistic(x=x, y=y))

my_pipeline = pipeline()

# %%
# Generate data
# -------------

def gen_data(n, p, true_beta):
    x = np.random.normal(loc = 0, scale = 1, size = (n, p))
    true_beta = true_beta.reshape(-1, 1)
    
    mu = x.dot(true_beta)
    Sigma = np.identity(n)
    Y = mu + np.random.normal(loc = 0, scale = 1, size = (n, 1))
    return x, Y, Sigma

x, y, sigma = gen_data(150, 5, np.asarray([0, 0, 0, 0, 0]))

# %%
# Run the pipeline
# -----------------

selected_features, p_values = my_pipeline([x, y], sigma)

print("Selected features: ", selected_features)
print("P-values: ", p_values)