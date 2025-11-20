## boston_crime_modelling.R
## Predicting crime rate (crim) using the Boston housing dataset
## Methods: full linear model, significance-based reduction, best subset,
##          forward and backward stepwise selection.

## 0. Packages & data ---------------------------------------------------------

library(MASS)   # for Boston
library(leaps)  # for regsubsets

data("Boston")
dim(Boston)     # 506 x 14

## Helper: test MSE -----------------------------------------------------------

mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

## 1. Train/test split --------------------------------------------------------

set.seed(1)
n <- nrow(Boston)

train_idx <- sample(1:n, n/2)        # indices for training rows
test_idx  <- setdiff(1:n, train_idx) # the remaining indices

train_dat <- Boston[train_idx, ]
test_dat  <- Boston[test_idx, ]

## 2. Full linear model using all predictors ----------------------------------

lm.full <- lm(crim ~ ., data = train_dat)
summary(lm.full)

full.pred <- predict(lm.full, newdata = test_dat)
full.mse  <- mse(test_dat$crim, full.pred)
full.mse   # ~ 41.55 in this split

## 3. Linear model using only significant predictors (p < 0.1) ----------------

coefs    <- summary(lm.full)$coefficients
sigcoefs <- coefs[coefs[, "Pr(>|t|)"] < 0.1, ]

# names of significant predictors (excluding intercept)
sig.vars <- setdiff(rownames(sigcoefs), "(Intercept)")

form.sig <- as.formula(
  paste("crim ~", paste(sig.vars, collapse = " + "))
)
form.sig  # should be something like: crim ~ dis + rad + ptratio + black + lstat + medv

lm.sig <- lm(form.sig, data = train_dat)
summary(lm.sig)

sig.pred <- predict(lm.sig, newdata = test_dat)
sig.mse  <- mse(test_dat$crim, sig.pred)
sig.mse   # ~ 42.43 in this split

## 4. Best subset selection (exhaustive) --------------------------------------

best.full <- regsubsets(crim ~ ., data = train_dat, nvmax = 13)
reg.summary <- summary(best.full)

which.min(reg.summary$cp)      # = 9 (by Cp)
which.min(reg.summary$bic)     # = 3 (by BIC)
which.max(reg.summary$adjr2)   # = 9 (by adj R^2)

## Model chosen by BIC (k = 3 predictors) -------------------------------------

k_bic <- which.min(reg.summary$bic)

best.coef.bic <- coef(best.full, id = k_bic)
vars.bic      <- names(best.coef.bic)[-1]  # drop intercept

form.bic <- as.formula(
  paste("crim ~", paste(vars.bic, collapse = " + "))
)
form.bic  # should be: crim ~ rad + black + lstat

lm.bic <- lm(form.bic, data = train_dat)
summary(lm.bic)

pred.bic <- predict(lm.bic, newdata = test_dat)
mse.bic  <- mse(test_dat$crim, pred.bic)
mse.bic  # ~ 42.41

## 5. Forward stepwise selection ----------------------------------------------

best.fwd <- regsubsets(crim ~ ., data = train_dat,
                       nvmax = 13, method = "forward")

reg.fwd <- summary(best.fwd)

which.min(reg.fwd$cp)      # 9
which.min(reg.fwd$bic)     # 3
which.max(reg.fwd$adjr2)   # 9

k_cp_fwd  <- which.min(reg.fwd$cp)
k_bic_fwd <- which.min(reg.fwd$bic)

## Forward: Cp model (9 predictors) -------------------------------------------

coef.cp.fwd <- coef(best.fwd, id = k_cp_fwd)
vars.cp.fwd <- names(coef.cp.fwd)[-1]

form.cp.fwd <- as.formula(
  paste("crim ~", paste(vars.cp.fwd, collapse = " + "))
)
form.cp.fwd

lm.cp.fwd <- lm(form.cp.fwd, data = train_dat)
summary(lm.cp.fwd)

pred.cp.fwd <- predict(lm.cp.fwd, newdata = test_dat)
mse.cp.fwd  <- mse(test_dat$crim, pred.cp.fwd)
mse.cp.fwd   # ~ 41.49

## Forward: BIC model (3 predictors) ------------------------------------------

coef.bic.fwd <- coef(best.fwd, id = k_bic_fwd)
vars.bic.fwd <- names(coef.bic.fwd)[-1]

form.bic.fwd <- as.formula(
  paste("crim ~", paste(vars.bic.fwd, collapse = " + "))
)
form.bic.fwd  # should match form.bic above

lm.bic.fwd <- lm(form.bic.fwd, data = train_dat)
summary(lm.bic.fwd)

pred.bic.fwd <- predict(lm.bic.fwd, newdata = test_dat)
mse.bic.fwd  <- mse(test_dat$crim, pred.bic.fwd)
mse.bic.fwd   # ~ 42.41

## 6. Backward stepwise selection ---------------------------------------------

best.bwd <- regsubsets(crim ~ ., data = train_dat,
                       nvmax = 13, method = "backward")

reg.bwd <- summary(best.bwd)

which.min(reg.bwd$cp)      # 9
which.min(reg.bwd$bic)     # 3
which.max(reg.bwd$adjr2)   # 9

k_cp_bwd  <- which.min(reg.bwd$cp)
k_bic_bwd <- which.min(reg.bwd$bic)

## Backward: Cp model (9 predictors) ------------------------------------------

coef.cp.bwd <- coef(best.bwd, id = k_cp_bwd)
vars.cp.bwd <- names(coef.cp.bwd)[-1]

form.cp.bwd <- as.formula(
  paste("crim ~", paste(vars.cp.bwd, collapse = " + "))
)
form.cp.bwd

lm.cp.bwd <- lm(form.cp.bwd, data = train_dat)
summary(lm.cp.bwd)

pred.cp.bwd <- predict(lm.cp.bwd, newdata = test_dat)
mse.cp.bwd  <- mse(test_dat$crim, pred.cp.bwd)
mse.cp.bwd   # ~ 41.49

## Backward: BIC model (3 predictors) -----------------------------------------

coef.bic.bwd <- coef(best.bwd, id = k_bic_bwd)
vars.bic.bwd <- names(coef.bic.bwd)[-1]

form.bic.bwd <- as.formula(
  paste("crim ~", paste(vars.bic.bwd, collapse = " + "))
)
form.bic.bwd  # should again match rad + black + lstat

lm.bic.bwd <- lm(form.bic.bwd, data = train_dat)
summary(lm.bic.bwd)

pred.bic.bwd <- predict(lm.bic.bwd, newdata = test_dat)
mse.bic.bwd  <- mse(test_dat$crim, pred.bic.bwd)
mse.bic.bwd   # ~ 42.41

#fit a lasso and ridge model 

#lasso model 
library(MASS)
library(glmnet)

data("Boston")

## 1. Build x and y -------------------------------------------------------

# model.matrix creates dummy variables etc.; drop the intercept column
x <- model.matrix(crim ~ ., Boston)[, -1]
y <- Boston$crim

## 2. Train/test split ----------------------------------------------------

set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test  <- (-train)
y.test <- y[test]

## 3. Fit lasso on a grid of lambda --------------------------------------

grid <- 10^seq(10, -2, length = 100)

# alpha = 1 for lasso (alpha = 0 would be ridge)
lasso.mod <- glmnet(x[train, ], y[train],
                    alpha = 1,
                    lambda = grid)

## 4. Cross-validation to choose best lambda ------------------------------

set.seed(1)
cv.out <- cv.glmnet(x[train, ], y[train],
                    alpha = 1,
                    lambda = grid)

bestlambda <- cv.out$lambda.min
bestlambda

## 5. Predict on test set & compute test MSE ------------------------------

lasso.pred <- predict(lasso.mod,
                      s = bestlambda,      # use the CV-chosen lambda
                      newx = x[test, ])

lasso.mse<-mean((as.numeric(lasso.pred) - y.test)^2)
lasso.mse # ~ 40.97

#fit lasso model 

## 1. Design matrix & response ---------------------------------------------

x <- model.matrix(crim ~ ., Boston)[, -1]  # drop intercept column
y <- Boston$crim

## 2. Train/test split -----------------------------------------------------

set.seed(1)
train <- sample(1:nrow(x), nrow(x) / 2)
test  <- (-train)
y.test <- y[test]

## 3. Lambda grid ----------------------------------------------------------

grid <- 10^seq(10, -2, length = 100)

## 4. Fit ridge model (alpha = 0) -----------------------------------------

ridge.mod <- glmnet(x[train, ], y[train],
                    alpha = 0,          # ridge
                    lambda = grid)

## 5. Cross-validation to choose best lambda -------------------------------

set.seed(1)
cv.ridge <- cv.glmnet(x[train, ], y[train],
                      alpha = 0,
                      lambda = grid)

bestlambda.ridge <- cv.ridge$lambda.min
bestlambda.ridge

## 6. Test MSE -------------------------------------------------------------

ridge.pred <- predict(ridge.mod,
                      s = bestlambda.ridge,
                      newx = x[test, ])

ridge.mse <- mean((as.numeric(ridge.pred) - y.test)^2)
ridge.mse # ~ 41.11

#fit using a PCR method
set.seed(1)
pcr.fit <- pcr(crim ~ .,
               data       = Boston,
               subset     = train_idx,
               scale      = TRUE,      # standardize predictors
               validation = "CV")      # K-fold CV

# Model summary: variance explained per component
summary(pcr.fit)

# Cross-validated MSEP plot (use this to choose ncomp)
validationplot(pcr.fit, val.type = "MSEP")

# From the plot, suppose the best trade-off is around 8 components:
best.ncomp <- 8


## 7.2 Test-set performance -----------------------------------------------

pcr.pred <- predict(pcr.fit,
                    newdata = Boston[test_idx, ],
                    ncomp   = best.ncomp)

pcr.mse <- mean((as.numeric(pcr.pred) - Boston$crim[test_idx])^2)
pcr.mse
#43.21097

#compare all models 

mse.summary <- c(
  full_ols        = full.mse,
  sig_ols         = sig.mse,
  best_subset_bic = mse.bic,
  fwd_cp          = mse.cp.fwd,
  fwd_bic         = mse.bic.fwd,
  bwd_cp          = mse.cp.bwd,
  bwd_bic         = mse.bic.bwd,
  lasso           = lasso.mse,
  ridge           = ridge.mse,
  pcr_8comp       = pcr.mse
)





