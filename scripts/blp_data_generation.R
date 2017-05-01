# This script generates data from a random coefficients logit model
library(evd)
library(rstan)
options(mc.cores = parallel::detectCores())

# compiled_blp <- stan_model("models/blp.stan")
# expose_stan_functions("models/blp.stan")









# Create data -------------------------------------------------------------


# Dimensions
J <- 10
T <- 20
NS <- 200
P <- 3
P2 <- 1
D <- 3

# Simulate demographics

Omega_dem <- matrix(c(1, .5, .5, .5 , 1, .5, .5, .5, 1), D, D)
Scale_dem <- c(1,2,3)
Dem <- MASS::mvrnorm(NS, c(0, 0, 0), diag(Scale_dem) %*% Omega_dem %*% diag(Scale_dem))

# Simulate individual shocks

v <- rnorm(NS)

# Product characteristics

x <- cbind(1, matrix(rnorm(J*(P-1)), J, P-1))
#x <- matrix(rnorm(J*(P)), J, P)
# Simulate true parameters
alpha <- -1*abs(rnorm(1))
beta <- rnorm(P)
Pi <- matrix(rnorm((P+1)*D), P+1, D)
Sigma <- matrix(rnorm(P+1))
gamma1 <- rnorm(1)
gamma2 <- rnorm(1)
inst_scale <- abs(rnorm(1))


# Simulate prices

# Instrument 
instrument <- matrix(rnorm(J*T), J*T, 1)

# Price
price <- gamma1 + gamma2 * instrument + rnorm(J*T, 0, inst_scale)

price_matrix <- matrix(price, T,J, byrow = T)

# Simulate structural shocks

Xi <- matrix(rnorm(J*T), T, J)

# Delta

delta <- matrix(NA, T, J)

for(i in 1:T) {
  delta[i,] = t(alpha * price_matrix[i,] + x %*% beta + Xi[i,])
}

# Simulate customer demographics

Dem_array <- array(NA,dim =  c(T,NS, D))
for(i in 1:T) {
  Dem_array[i,,] <- Dem
}


# Simulate customer utility for each market

customer_draws <- array(NA, dim = c(T, NS, J))

market_shares <- matrix(NA, T, J)
  
for(i in 1:T) {
  random_shocks <-  Dem_array[i,,] %*% t(Pi) +  v %*% t(Sigma)
  del <- matrix(rep(delta[i,], NS), NS, J, byrow = T)
  for(j in 1:J) {
    customer_draws[i,,j] <-  del[,j] + random_shocks %*% c(price_matrix[i,j], x[j,]) + rgumbel(NS, loc = -.5772)
  }
  customer_draws[i,,] <- exp(customer_draws[i,,]) / matrix(rep(rowSums(exp(customer_draws[i,,])), J), NS, J)
  market_shares[i,] <- colMeans(customer_draws[i,,])
}


market_shares_est <- market_shares[,-J]

J_est <- J-1
price_est <- as.vector(t(price_matrix[,-J]))
instrument_est <- as.vector(t(matrix(instrument, T, J, byrow = T)[,-J]))
x_est <- x[-J,]


# Testing functions -------------------------------------------------------


# mu_it(x_t = x_est, p_t = price_est[1:(J-1)], D_i = Dem[1,], v_i = v[1], Pi = Pi, Sigma = Sigma)
# 
# s1 <- shares(x_t = x_est, p_t = price_est[1:(J-1)], D = Dem, v = v, Pi = Pi, Sigma = Sigma, delta_t = log(market_shares[1,-J]) - log(1 - sum(market_shares[1,-J])))
# s1
# sum(s1)
# delta_1(observed_shares = market_shares_est[1,], x_t = x_est, p_t = price_est[1:(J-1)], D = Dem, v = v, Pi = Pi, Sigma = Sigma, tol = 1e-8)
# 
# while(diff>1e-8){
#   s2 <- shares(x_t = x_est, p_t = price_est[1:(J-1)], D = Dem, v = v, Pi = Pi, Sigma = Sigma, delta_t = init)
#   init2 <- init + log(matrix(market_shares_est[1,])) - log(t(s2))
#   diff <- crossprod(init2 - init)
#   init <- init2
# }
# init

# Run model ---------------------------------------------------------------


compiled_blp_dirichlet <- stan_model("models/blp_dirichlet2.stan")



test_est_2 <- optimizing(compiled_blp_dirichlet, data = list(J = J_est,
                                               T = T,
                                               NS = NS,
                                               P = P,
                                               P2 = P2,
                                               D = D,
                                               market_shares = market_shares,
                                               price = price_est,
                                               instruments = matrix(instrument_est),
                                               x = x_est,
                                               Dem = Dem_array,
                                               lambda = exp(3),
                                               lambda2 = 3)), cores = 4, iter = 200,refresh= 10)



pars <- get_posterior_mean(test_est)[,1]

pars <- test_est_2$par
alpha
pars[grep("alpha", names(pars))]
beta
pars[grep("beta", names(pars))]
gamma1
pars[grep("gamma1", names(pars))]
gamma2
pars[grep("gamma2", names(pars))]

test_est_2$par[1]

a <- as.vector(matrix(pars[grep("predicted_shares", names(pars))], 20, 10))
b <- as.vector(market_shares)
plot(scale(b - a), as.vector(Xi))
plot(as.vector(pars[grep(x = names(pars),"Xi")]), as.vector(Xi[,-J]))
plot(as.vector(test_est_2$par[grep(x = names(test_est_2$par),"Xi")]), as.vector(Xi[,-J]))
matrix(test_est$par[grep(x = names(test_est$par),"original_shares")], 20, 7)
matrix(test_est$par[grep(x = names(test_est$par),"predicted_shares")], 20, 8)
true_pars <- c(alpha, beta, Pi, Sigma, gamma1, gamma2, inst_scale)
par_estimates <- pars[2:24]

plot(data.frame(true_pars, par_estimates))
 abline(a = 0, b = 1)
