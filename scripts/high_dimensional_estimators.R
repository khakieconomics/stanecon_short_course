# This script illustrates the properties of techniques to build models where inputs/covariates 
# are high-dimensioned categorical variables. 

# Load libraries

library(reshape2); library(ggplot2); library(rstan); library(glmnet); library(dplyr); library(rstanarm)
options(mc.cores = parallel::detectCores())
# The generative model

# The proposed generative of individual i in group j is y_ij ~ normal(mu_j, sigma) where the number of
# groups is large. mu_j ~ normal(mu, tau)

generate_data <- function(J, # Number of groups 
                          N, # Number of observations
                          mu, # Mean of group effects
                          tau, # scale of group effects
                          sigma # Scale of individual errors
                          ) {
  # mu_j
  group_effects <- data_frame(j = as.character(1:J), mu_j = rnorm(J, mu, tau))
  
  # Individual_model
  
  # group membership probabilities
  probs <- arm::invlogit(rnorm(J))
  probs <- probs/sum(probs)
  
  # Generate individual data
  individuals <- data_frame(individual = 1:N, 
                            individual_error = rnorm(N, 0, sigma),
                            j = as.character(sample(1:J, N, replace = T, prob = probs))) %>% 
    left_join(group_effects, by = "j") %>% 
    mutate(y = mu_j + individual_error)
  
  # Return individuals
  individuals
}



# An experiment ----

# Generate data

fake_data <- generate_data(1500, 2000, mu = 3, tau = 3, sigma = 5)

# Estimate with various methods

# Maximum likelihood random effects
ml_model <- lme4::lmer(y ~ 1 + (1 | j), data= fake_data)

ml_ranefs <- ranef(ml_model, condVar = T)[[1]]
ml_ranefs$sd <- sqrt((attr(ml_ranefs, "postVar") %>%  plyr::adply(3))[,2])

estimates_ml <- data_frame(ml_est = ranef(ml_model)[[1]][,1] + fixef(ml_model),
                           ml_lower = ml_est - 1.64*ml_ranefs$sd,
                           ml_upper = ml_est + 1.64*ml_ranefs$sd,
                           j = rownames(ranef(ml_model)[[1]]))

# OLS
ols_model <- lm(y ~ -1 + j, data = fake_data)
estimates_ols <- data_frame(ols_est = coef(ols_model),
                            j = gsub("j", "", names(coef(ols_model))))

# LASSO
x <- model.matrix(~ -1 + j, data = fake_data)
lasso_mod <- cv.glmnet(x = x, y = fake_data$y, family = "gaussian", alpha = 1)
estimates_lasso <- data_frame(lasso_est = coef(lasso_mod, s = "lambda.min") %>% as.numeric,
                              j = gsub("j", "", rownames(coef(lasso_mod, d = "lambda.min"))))

# Ridge
ridge_mod <- cv.glmnet(x = x, y = fake_data$y, family = "gaussian", alpha = 0)
estimates_ridge <- data_frame(ridge_est = coef(ridge_mod, s = "lambda.min") %>% as.numeric,
                              j = gsub("j", "", rownames(coef(ridge_mod, d = "lambda.min"))))


# Bayesian random effects
compiled_model <- stan_model("models/ch_1/simple_varying_intercepts.stan")
bayes_mod2 <- sampling(compiled_model, data = list(N = nrow(fake_data), 
                                                   J = length(unique(fake_data$j)),
                                                   group = as.numeric(as.factor(fake_data$j)),
                                                   Y = fake_data$y), cores = 4)

bayes_means <- rstan::get_posterior_mean(bayes_mod2, par = "mu_j")[,5]

draws <- extract(bayes_mod2, pars = "mu_j", permuted = T)[[1]] %>% as.data.frame

lower <- function(x) quantile(x, 0.05)
upper <- function(x) quantile(x, 0.95)

bayes_mean <- draws %>% as.data.frame %>% 
  summarise_all(.funs = funs(mean))  %>% unlist()

bayes_lower <- draws %>% as.data.frame %>%
  summarise_all(.funs = funs(lower))  %>% unlist()

bayes_upper <- draws %>% as.data.frame %>%
  summarise_all(.funs = funs(upper))  %>% unlist()

ranefs <- bayes_mean 



estimates_bayes <- data_frame(bayes_predict = predict(bayes_mod),
                              bayes_est = ranefs,
                              bayes_lower = bayes_lower , 
                              bayes_upper = bayes_upper,
                              j = levels(as.factor(fake_data$j)))

effect_estimates <- estimates_ml %>% 
  left_join(estimates_ols, by = "j") %>% 
  left_join(estimates_lasso, by = "j") %>% 
  left_join(estimates_ridge, by = "j") %>% 
  left_join(estimates_bayes, by = "j") %>% 
  left_join(fake_data %>% group_by(j) %>% summarise(true_value = first(mu_j)), by = "j") %>% 
  select(-j)

cor(effect_estimates)
effect_estimates %>% 
  select(-contains("lower"), -contains("upper")) %>% 
  melt(id = "true_value") %>%
  ggplot(aes(x = value, colour = variable)) +
  stat_ecdf() +
  stat_ecdf(aes(x = true_value), colour = "black")


effect_estimates %>% 
  arrange(ml_est) %>% 
  ggplot(aes(x = 1:nrow(effect_estimates), y = bayes_est)) +
  geom_linerange(aes(ymin = bayes_lower, ymax = bayes_upper), colour = "orange") +
  geom_line() +
  geom_point(aes(y = true_value), alpha =0.5)

effect_estimates %>% 
  arrange(ml_est) %>% 
  ggplot(aes(x = 1:nrow(effect_estimates), y = ml_est)) +
  geom_linerange(aes(ymin = ml_lower, ymax = ml_upper), colour = "orange") +
  geom_line() +
  geom_point(aes(y = true_value), alpha =0.5)

effect_estimates %>% 
  summarise(coverage_bayes = mean(true_value < bayes_upper & true_value > bayes_lower),
            coverage_ml = mean(true_value < ml_upper & true_value > ml_lower))

