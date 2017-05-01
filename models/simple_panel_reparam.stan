data {
  int N; // number of observations in total
  int J; // number of technologies
  vector[N] time; // time 
  int tech[N]; // tech index
  vector[N] y; // the log levels of the technology
}
parameters {
  matrix[J, 3] z;
  vector[3] theta_mu;
  vector<lower = 0>[3] theta_tau;
  corr_matrix[3] Omega;
}
transformed parameters {
  matrix[J, 3] theta;
  for(j in 1:J) {
    theta[j] = (theta_mu + cholesky_decompose(quad_form_diag(Omega, theta_tau)) * z[j]')';
  }
}
model {
  theta_mu ~ normal(0, 1);
  theta_tau ~ cauchy(0, 1);
  Omega ~ lkj_corr(2);
  
  to_vector(z) ~ normal(0, 1);
  
  for(i in 1:N) {
    y[i] ~ normal(theta[tech[i], 1] + theta[tech[i], 2]* time[i], exp(theta[tech[i], 3]));
  }
}