### `boston_crime_modelling.R`

In this script, I model the per-capita crime rate (`crim`) in the classic Boston housing dataset (`MASS::Boston`). I use this as a small playground to compare different linear modelling and regularization techniques.

#### What I do

1. **Prepare the data**
   - Load `MASS::Boston` and keep `crim` as the response.
   - Create a 50/50 train–test split with `set.seed(1)` for reproducibility.
   - Define a helper `mse()` function for test mean squared error.

2. **Baseline linear models**
   - **Full OLS model**: `crim ~ .` using all predictors.
   - **Significance-based reduced OLS**: refit a model keeping only predictors with `p < 0.1` from the full model.
   - Evaluate both models on the test set via MSE.

3. **Subset and stepwise selection (using `leaps::regsubsets`)**
   - **Best subset selection** (exhaustive search) for models with up to 13 predictors.
   - Use **Cp**, **BIC**, and **adjusted R²** to choose candidate models:
     - Best subset BIC model (usually 3 predictors, e.g. `rad + black + lstat`).
   - **Forward stepwise selection**:
     - Cp-selected model (9 predictors).
     - BIC-selected model (3 predictors).
   - **Backward stepwise selection**:
     - Cp-selected model (9 predictors).
     - BIC-selected model (3 predictors).
   - For each chosen model, I refit it with `lm()` on the training data and compute test MSE.

4. **Regularized regression (using `glmnet`)**
   - Build a design matrix with `model.matrix(crim ~ ., Boston)[, -1]` for lasso/ridge.
   - Use a logarithmic grid of penalty values: `lambda = 10^seq(10, -2, length = 100)`.
   - **Lasso regression** (`alpha = 1`):
     - Perform `cv.glmnet()` on the training set.
     - Use `lambda.min` from cross-validation.
     - Predict on the test set and compute test MSE.
   - **Ridge regression** (`alpha = 0`):
     - Same CV pipeline as lasso.
     - Predict on the test set and compute test MSE.

5. **Principal Component Regression (PCR, using `pls`)**
   - Fit `pcr(crim ~ ., data = Boston, subset = train_idx, scale = TRUE, validation = "CV")`.
   - Inspect the cross-validated MSEP curve with `validationplot()` to choose the number of components.
   - In this run, I pick **8 principal components** as a reasonable trade-off.
   - Predict on the test set and compute test MSE.

#### Key results (test MSE, 50/50 split, `set.seed(1)`)

The exact values will depend slightly on the split, but in this configuration I get approximately:

- **Full OLS**: ~41.6  
- **Reduced OLS (p < 0.1)**: ~42.4  
- **Best subset (BIC, 3 predictors)**: ~42.4  
- **Forward stepwise (Cp, 9 predictors)**: ~41.5  
- **Backward stepwise (Cp, 9 predictors)**: ~41.5  
- **Lasso (CV-selected λ)**: ~41.0  
- **Ridge (CV-selected λ)**: ~41.1  
- **PCR (8 components)**: ~43.2  

#### Conclusion

On this train–test split, the **lasso model** gives the **lowest test MSE** (around 41), slightly beating the best stepwise and ridge models, and clearly outperforming the reduced OLS and PCR models. The differences are small and all linear/penalized models are in a similar ballpark, but if I had to pick one model for out-of-sample prediction of `crim` in this setup, **I would choose lasso with cross-validated λ**.
