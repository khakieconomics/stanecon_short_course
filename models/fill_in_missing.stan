data {
  int T;
  vector[T] Y;
  int n_missing;
}
parameters {
  real a;
  real b;
  real<lower = 0> sigma;
  vector[n_missing] missing;
}
transformed parameters {
  vector[T] Y2;
  {
    int count;
    count = 0;
    for(t in 1:T) {
      if(Y[t]==-9) {
        count = count + 1;
        Y2[t] = missing[count];
      } else {
        Y2[t] = Y[t];
      }
    }
  }
  
}
model {
  a ~ normal(0, 1);
  b ~ normal(0, 1);
  sigma ~ student_t(4, 0 , 1);
  
  // likelihood
  for(t in 2:T) {
    Y2[t] ~ normal(a + b*Y2[t-1], sigma);
  }
}
