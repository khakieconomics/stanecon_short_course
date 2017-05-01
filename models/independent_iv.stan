// saved as models/independent_iv.stan
data {
  int N; // number of observations
  int P; // number of covariates
  matrix[N, P] X; //covariate matrix
  vector[N] Y; //outcome vector
  vector[N] endog_regressor; // the endogenous regressor
  vector[N] Z; // the instrument (which we'll assume is a vector)
}
parameters {
  vector[P] beta; // the regression coefficients
  vector[P] gamma;
  real tau;
  real delta;
  real alpha_1;
  real alpha_2;
  vector<lower = 0>[2] sigma; // the residual standard deviation
  corr_matrix[2] Omega;
}
transformed parameters {
  matrix[N, 2] mu;
  
  for(i in 1:N) {
    mu[i,2] = alpha_1 + X[i]*gamma + Z[i]*delta;
    mu[i,1] = alpha_2 + X[i]*beta + mu[i,2]*tau;
  }
}
model {
  // Define the priors
  beta ~ normal(0, 1); 
  gamma ~ normal(0, 1);
  tau ~ normal(0, 1);
  sigma ~ cauchy(0, 1);
  delta ~ normal(0, 1);
  alpha_1 ~ normal(0, 1);
  alpha_2 ~ normal(0, 2);
  Omega ~ lkj_corr(5);
  
  // The likelihood
  for(i in 1:N) {
    Y[i]~ normal(mu[i], sigma[1]);
    endog_regressor[i]~ normal(mu[2], sigma[2]);
  }

  
}
generated quantities {
  // For model comparison, we'll want to keep the likelihood
  // contribution of each point

  vector[N] log_lik;
  for(i in 1:N) {
    log_lik[i] = normal_lpdf(Y[i] | alpha_1  + X[i,] * beta + endog_regressor[i]*tau, sigma[1]);
  }
}
