functions {
    // Define the ODE systems or any other functions needed
    vector growth_model(real t, vector y, vector params) {
        vector[1] dydt;
        real s = params[1];     // constant source term
        real b = params[2];     // division rate
        real d = params[3];     // loss rate

        dydt[1] = s + b * y[1] - d * y[1];     // growth model with source term 
        return dydt;
    }

    vector Tx_model(real t, vector y, vector params) {
        vector[1] dydt;
        real s = params[1];                 // source term
        real b = params[2];     // division rate
        real d = params[3];     // loss rate

        real tx_time = 21.0; // time at which treatment starts
        if (t >= tx_time) {
            s = 0.0; // source term is zero after treatment
        } else {
            s = params[1]; // source term before treatment
        }

        dydt[1] = b * y[1] - d * y[1];     // tx model without source term 
        return dydt;
    }
}

data {
    int<lower=1> T1;                   // Number of time points in dataset 1
    int<lower=1> T2;                   // Number of time points in dataset 2
    real t0;                           // Initial time
    real y0;                           // Initial condition (cell count at time t0)
    array[T1] real time_obs1;          // Observation times dataset 1
    array[T1] real y_obs1;             // Observed values dataset 1
    array[T2] real time_obs2;          // Observation times dataset 2
    array[T2] real y_obs2;             // Observed values dataset 2
}

parameters {
    real<lower=0> s;          // constant source term
    real<lower=0> b;          // division rate
    real<lower=b> d;          // loss rate
    real<lower=0> sigma;      // Observation noise
}

model {
    // Priors on model parameters -- this is where we encode our previous befliefs about the system in the model
    s ~ normal(1e5, 3e4);
    b ~ normal(0.3, 0.25);
    d ~ normal(0.6, 0.25);
    sigma ~ normal(0.5, 0.5);            // noise in the data -- we estimate this too! pretty cool!

    vector[1] y_init = [y0]';          // define the Initial state as a vector to comply with the ode solver signature
    vector[3] params = [s, b, d]';     // parameters as a vector to comply with the ode function

    // transform y_obs to log scale -- data tranformations are helpful and are often necessary -- your data may not always be on alwyas linear scale
    array[T1] real log_y_obs1;
    array[T2] real log_y_obs2;

    for (t in 1:T1) {
        log_y_obs1[t] = log(y_obs1[t]);    
    }

    for (t in 1:T2) {
        log_y_obs2[t] = log(y_obs2[t]);    
    }
    
    // Solve the ODEs
    array[T1] vector[1] y_hat1 = ode_rk45(growth_model, y_init, t0, time_obs1, params);
    array[T2] vector[1] y_hat2 = ode_rk45(Tx_model, y_init, t0, time_obs2, params);

    // Likelihood --- here we assume our data is normally distributed
    for (t in 1:T1) {
        log_y_obs1[t] ~ normal(log(y_hat1[t, 1]), sigma);
    }

    for (t in 1:T2) {
        log_y_obs2[t] ~ normal(log(y_hat2[t, 1]), sigma);
    }
}
