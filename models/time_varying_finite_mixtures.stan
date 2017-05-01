// saved as time_varying_finite_mixtures.stan
data {
  int T;
  vector[T] r;
}
parameters {
  vector[T] mu;
  vector[2] rho;
  real beta;
  vector<lower = 0>[3] sigma;
  vector[3] alpha; 
}
model {
  // priors 
  mu[1] ~ normal(0, .1);
  sigma ~ cauchy(0, 0.5);
  rho ~ normal(1, .1);
  beta~ normal(.5, .25);
  alpha[1:2] ~ normal(0, 0.1);
  alpha[3] ~ normal(0, 1);

  // likelihood
  for(t in 2:T) {
    mu[t] ~ normal(alpha[3] + rho[1]*mu[t-1] + beta* r[t-1], sigma[3]);

    target += log_mix(inv_logit(mu[t]), 
                      normal_lpdf(r[t] | alpha[1], sigma[1]), 
                      normal_lpdf(r[t] | alpha[2] + rho[2] * r[t-1], sigma[2]));
  }
}