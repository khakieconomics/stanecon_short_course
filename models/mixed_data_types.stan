data {
  int N;
  int P; // number of continuous outcomes
  int nber[N]; // indicator for NBER recession
  matrix[N, P] Y; // observed continuous outcomes
}
parameters {
  vector[N] latent_series;
  vector[P+1] mu;
  vector<lower = 0>[P] scale;
  corr_matrix[P+1] Omega;
}
transformed parameters{
  matrix[N, P+1] Y2;
  vector[P+1] scales;
  Y2 = append_col(Y, latent_series);
  scales[1:2] = scale;
  scales[3] = 1.0;
}
model {
  mu ~ normal(0, 1);
  scale ~ student_t(4, 0, 1);
  Omega ~ lkj_corr(4);
  
  for(n in 1:N) {
    Y2[n] ~ multi_normal(mu, quad_form_diag(Omega, scales));
    nber[n] ~ bernoulli(Phi_approx(latent_series[n]));
  }
}