functions {
    // Define the ODE system 
    vector exponential_decay(real t, vector y, vector params) {
        vector[1] dydt;
        real s = params[1];     // constant source term
        real b = params[2];     // division rate
        real d = params[3];     // loss rate

        dydt[1] = s + b * y[1] - d * y[1];     //ode 
        return dydt;
    }
}

data {
    int<lower=1> T;                    // Number of time points
    real t0;                           // Initial time
    real y0;                           // Initial condition (cell count at time t0)
    array[T] real time_obs;            // Observation times
    array[T] real y_obs;               // Observed values
}

parameters {
    real<lower=0> s;          // constant source term
    real<lower=0> b;          // division rate
    real<lower=b> d;          // loss rate
    real<lower=0> sigma;      // Observation noise
}

model {
    // Priors on model parameters -- this is where we encode our previous befliefs about the system in the model
    s ~ normal(8e4, 3e3);
    b ~ normal(0.3, 0.25);
    d ~ normal(0.6, 0.25);
    sigma ~ normal(0.5, 0.5);            // noise in the data -- we estimate this too! pretty cool!

    vector[1] y_init = [y0]';          // define the Initial state as a vector to comply with the ode solver signature
    vector[3] params = [s, b, d]';     // parameters as a vector to comply with the ode function

    // transform y_obs to log scale -- data tranformations are helpful and are often necessary -- your data may not always be on alwyas linear scale
    array[T] real log_y_obs;
    for (t in 1:T) {
        log_y_obs[t] = log(y_obs[t]);    
    }

    // Solve the ODE
    array[T] vector[1] y_hat = ode_rk45(exponential_decay, y_init, t0, time_obs, params);
    
    // Likelihood --- here we assume our data is normally distributed
    for (t in 1:T) {
        log_y_obs[t] ~ normal(log(y_hat[t, 1]), sigma);
    }
}
