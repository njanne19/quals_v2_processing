import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.integrate import solve_ivp
from functools import partial


timeline = np.arange(0, 0.3, 0.001)


# Generate a 1st order 
alpha = 20
phi_heavy_true = pd.DataFrame({
    'time': timeline,
    'phi': 1 - np.exp(-alpha*timeline)
})
phi_heavy_true_dot = pd.DataFrame({
    'time': timeline, 
    'phi_dot': alpha*np.exp(-alpha*timeline)
})

# # Plot the true phi_heavy
# plt.plot(timeline, phi_heavy_true['phi'], label='True phi_heavy')
# plt.legend()
# plt.show()


# Then we need to propagate the forward dynamics 
def forward_dynamics(t, y, phis, impedance, load_func):
    """
    Forward dynamics of the system. 

    theta and its derivatives are propagated. This function specifically returns theta_ddot. 
    phi is the reference motion source behind the impedance given by impedance. 
    
    Impedance is the impedance of the system, given as [J, b, k]. 
    load_func is the load function of the system, and is a function of theta. 
    
    """

    J = impedance[0]
    b = impedance[1]
    k = impedance[2]

    theta = y[0]
    theta_dot = y[1]

    phi = phis[0]['phi'][np.argmin(np.abs(phis[0]['time'] - t))]
    phi_dot = phis[1]['phi_dot'][np.argmin(np.abs(phis[1]['time'] - t))]

    theta_ddot = (1/J) * ((phi - theta)*k + (phi_dot - theta_dot)*b - load_func(theta))

    # print(f'phi: {phi}, phi_dot: {phi_dot}', end='\t')
    # print(f'theta: {theta}, theta_dot: {theta_dot}, theta_ddot: {theta_ddot}')

    return [theta_dot, theta_ddot]

def inverse_dynamics(t, y, thetas, impedance, load_func): 
    """
    Inverse dynamics of the system. Calculates the control input phi for a given set of states, 
    alongside impedances and load functions. 
    """
    theta = thetas[0]['theta'][np.argmin(np.abs(thetas[0]['time'] - t))]
    theta_dot = thetas[1]['theta_dot'][np.argmin(np.abs(thetas[1]['time'] - t))]
    theta_ddot = thetas[2]['theta_ddot'][np.argmin(np.abs(thetas[2]['time'] - t))]

    J = impedance[0]
    b = impedance[1]
    k = impedance[2]

    phi = y[0]
    phi_dot = -1 * (k/b) * phi + (1/b) * (k * theta + b * theta_dot + J*theta_ddot + load_func(theta))

    return [phi_dot]

# Define the heavy and light load functions 
def heavy_load_func(theta):
    return np.min([0.4, 0.4*(theta/0.1)])

def light_load_func(theta): 
    return np.min([0.1, 0.1*(theta/0.1)])

##########################################################################
# GROUND TRUTH 
##########################################################################
# Now let's set the ground truth impedance 
impedance_heavy_true = np.array([6e-4, 0.05, 2.0])
impedance_light_true = np.array([6e-4, 0.05, 0.5])

# Now let's propagate the forward dynamics 
forward_dynamics_heavy_true = partial(
    forward_dynamics, 
    phis=[phi_heavy_true, phi_heavy_true_dot], 
    impedance=impedance_heavy_true, 
    load_func=heavy_load_func
)
forward_dynamics_heavy_light_true = partial(
    forward_dynamics,
    phis=[phi_heavy_true, phi_heavy_true_dot], 
    impedance=impedance_heavy_true, 
    load_func=light_load_func
)
theta_heavy_true = solve_ivp(forward_dynamics_heavy_true, t_span=[0, 1], t_eval=timeline, y0=[0, 0], method='RK45')
theta_heavy_light_true = solve_ivp(forward_dynamics_heavy_light_true, t_span=[0, 1], t_eval=timeline, y0=[0, 0], method='RK45')

# Package the results of theta_heavy_true into an array of dataframes with time as well 
theta_heavy_true_df = pd.DataFrame({
    'time': timeline,
    'theta': theta_heavy_true.y[0],
})
theta_heavy_true_dot_df = pd.DataFrame({
    'time': timeline,
    'theta_dot': theta_heavy_true.y[1]
})
# Then we need to calcaulate the second derivative of theta (just passing through forward_dynamics_heavy_true)
theta_heavy_true_ddot_df = pd.DataFrame({
    'time': timeline,
    'theta_ddot': [forward_dynamics_heavy_true(t, y)[1] for t, y in zip(timeline, theta_heavy_true.y.T)]
})
thetas_heavy_true = [theta_heavy_true_df, theta_heavy_true_dot_df, theta_heavy_true_ddot_df]

# Now let's do the same for the heavy-light catch case 
theta_heavy_light_true_df = pd.DataFrame({
    'time': timeline,
    'theta': theta_heavy_light_true.y[0]
})
theta_heavy_light_true_dot_df = pd.DataFrame({
    'time': timeline,
    'theta_dot': theta_heavy_light_true.y[1]
})
theta_heavy_light_true_ddot_df = pd.DataFrame({
    'time': timeline,
    'theta_ddot': [forward_dynamics_heavy_light_true(t, y)[1] for t, y in zip(timeline, theta_heavy_light_true.y.T)]
})
thetas_heavy_light_true = [theta_heavy_light_true_df, theta_heavy_light_true_dot_df, theta_heavy_light_true_ddot_df]
##########################################################################
##########################################################################
# INDIVIDUAL IMPEDANCE 
##########################################################################
# Now let's propagate the inverse dynamics to get a sloppy estimate for phi 
def guess_impedance_with_error(impedance_true, error_percentage): 
    # Sample a gaussian random variable with mean 0 and std of error_percentage * impedance_true 
    error = np.random.normal(0, error_percentage * impedance_true)
    return impedance_true + error

impedance_heavy_individual = guess_impedance_with_error(impedance_heavy_true, 0.20)
inverse_dynamics_heavy_individual = partial(
    inverse_dynamics, 
    thetas=thetas_heavy_true, 
    impedance=impedance_heavy_individual, 
    load_func=heavy_load_func
)
phi_heavy_individual = solve_ivp(inverse_dynamics_heavy_individual, t_span=[0, 1], t_eval=timeline, y0=[0], method='RK45')
phi_heavy_individual_df = pd.DataFrame({
    'time': timeline,
    'phi': phi_heavy_individual.y[0]
})
phi_dot_heavy_individual_df = pd.DataFrame({
    'time': timeline,
    'phi_dot': [inverse_dynamics_heavy_individual(t, y)[0] for t, y in zip(timeline, phi_heavy_individual.y.T)]
})
phis_heavy_individual = [phi_heavy_individual_df, phi_dot_heavy_individual_df]

# Now propagate the catch case 
forward_dynamics_heavy_light_individual = partial(
    forward_dynamics,
    phis=phis_heavy_individual, 
    impedance=impedance_heavy_individual, 
    load_func=light_load_func
)
theta_heavy_light_individual = solve_ivp(forward_dynamics_heavy_light_individual, t_span=[0, 1], t_eval=timeline, y0=[0, 0], method='RK45')

theta_heavy_light_individual_df = pd.DataFrame({
    'time': timeline,
    'theta': theta_heavy_light_individual.y[0]
})
theta_heavy_light_individual_dot_df = pd.DataFrame({
    'time': timeline,
    'theta_dot': theta_heavy_light_individual.y[1]
})
theta_heavy_light_individual_ddot_df = pd.DataFrame({
    'time': timeline,
    'theta_ddot': [forward_dynamics_heavy_light_individual(t, y)[1] for t, y in zip(timeline, theta_heavy_light_individual.y.T)]
})
thetas_heavy_light_individual = [theta_heavy_light_individual_df, theta_heavy_light_individual_dot_df, theta_heavy_light_individual_ddot_df]
##########################################################################
##########################################################################
# PLOTTING 
##########################################################################
# Plot theta and its derivatives alongside phi true 
plt.plot(timeline, thetas_heavy_light_true[0]['theta'], label='True Catch Case', color='black')
plt.plot(timeline, thetas_heavy_light_individual[0]['theta'], label='Predicted Impedance Catch Case', linestyle='--', color='red')
plt.plot(timeline, phi_heavy_true['phi'], label='True phi_heavy', linestyle=':', color='black')
plt.plot(timeline, phis_heavy_individual[0]['phi'], label='Recovered phi_heavy', linestyle=':', color='red')
plt.legend()
plt.grid()
plt.show()

