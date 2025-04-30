import pandas as pd
import numpy as np 
import json
from scipy.signal import savgol_filter 
from scipy.integrate import cumulative_trapezoid, solve_ivp
from TrialData import CalibrationTrial, MainTrial
from typing import List
import matplotlib.pyplot as plt
load_map = {
    0.1 : 'very_light', 
    0.2 : 'light', 
    0.3 : 'medium', 
    0.4 : 'moderately_heavy', 
    0.5 : 'heavy', 
}

def split_runs_into_trial_types(runs: dict) -> dict:
    """
    Split the trials into light, heavy, light->heavy, and heavy->light.
    Only returns valid trials.
    """
    light_trials = []
    heavy_trials = []
    light_heavy_trials = []
    heavy_light_trials = []
    
    invalid_trials = {}
    for run_idx, run in runs.items():
        for trial_idx, trial in run['trials'].items():
            trial_data = trial['data']
            trial_type = None
            
            if trial['load'] == 'light' or trial['load'] == 'very_light':
                if trial['is_catch']:
                    trial_type = 'heavy_light'
                else:
                    trial_type = 'light'
            elif trial['load'] == 'moderately_heavy' or trial['load'] == 'heavy' or trial['load'] == 'medium':
                if trial['is_catch']:
                    trial_type = 'light_heavy'
                else:
                    trial_type = 'heavy'
                    
            if trial['is_valid']:
                if trial_type == 'light':
                    light_trials.append(trial_data)
                elif trial_type == 'heavy':
                    heavy_trials.append(trial_data)
                elif trial_type == 'light_heavy':
                    light_heavy_trials.append(trial_data)
                elif trial_type == 'heavy_light':
                    heavy_light_trials.append(trial_data)
            else:
                if trial_type not in invalid_trials:
                    invalid_trials[trial_type] = 1
                else: 
                    invalid_trials[trial_type] += 1
    
    results = {
        'light': light_trials,
        'heavy': heavy_trials,
        'light_heavy': light_heavy_trials,
        'heavy_light': heavy_light_trials,
        'invalid_trials': invalid_trials
    }
    
    return results

def estimate_average_trajectories_single_participant(results_dict, task: str, use_corrected_outliers: bool = False): 
    """
    Estimate the average trajectories for a single participant. 
    """
    if task == 'calibration': 
        raise ValueError("Calibration task does not need average trajectory estimation.")
    
    elif task == 'main': 
        # First we get the average trial from all the non-outlier trials. 
        trajectory_dict = {}
        for run in results_dict['main']['runs']: 
            # Skip any uncorrected outliers
            if run.is_outlier() and not run.is_corrected_outlier():
                continue
            
            # If we're using corrected outliers, include them; otherwise skip all outliers
            if use_corrected_outliers or not run.is_outlier():
                load = run.get_load()
                catch_trial = run.is_catch_trial()
                loads_visible = run.is_load_visible()
                # Create a composite key that includes load, catch trial status, and visibility
                if catch_trial and loads_visible and load == 0.3: 
                    print(f"Found first medium catch trial with visible load.") 
                    key = 'first_medium_catch_visible'
                else: 
                    key = f"{load}_{'catch' if catch_trial else 'normal'}_{'visible' if loads_visible else 'hidden'}"
                if key not in trajectory_dict: 
                    trajectory_dict[key] = []
                trajectory_dict[key].append(run)
                
        # Now we average the trajectories. 
        for key, trials in trajectory_dict.items(): 
            # Each trial is a MainTrial object which has an underlying pd.DataFrame.
            # We need to do the following steps: 
            # 1. Align all time indices. Some trials will have more data than others, though they all start at the same time. 
            # 2. Average the data for each time index. For trials with missing data points, we will interpolate. 
            # 3. Add the average trial to the results dict. 
            
            # First, align the time indices. We should find the minimum set containing all the time indices.
            # Then, we will interpolate the missing data points. 
            # Flatten all elapsed time values from all trials into a single array and get unique values
            elapsed_time_values = np.concatenate([trial['elapsed_time'].values for trial in trials])
            unique_elapsed_time_values = np.unique(elapsed_time_values)
            unique_elapsed_time_values.sort()
            
            average_trial = pd.DataFrame() 
            average_trial['elapsed_time'] = unique_elapsed_time_values
            
            # Now, we will interpolate the missing data points and collect interpolated values
            columns_to_average = ['EncoderRadians_smooth', 'EncoderRadians_dot_smooth', 
                                 'EncoderRadians_ddot_smooth', 'GripForce', 'GripForce_dot_smooth']
            optional_columns = ['CommandTorque', 'WallTorque', 'VirtualSpringTorque', 'AppliedTorque']
            
            # Initialize dictionaries to collect interpolated values for each column
            interpolated_values = {col: [] for col in columns_to_average + optional_columns}
            
            for trial in trials:
                # Interpolate required columns and collect values
                for col in columns_to_average:
                    interpolated = np.interp(unique_elapsed_time_values, trial['elapsed_time'], trial[col])
                    interpolated_values[col].append(interpolated)
                
                # Interpolate optional columns if they exist
                for col in optional_columns:
                    if col in trial.columns:
                        interpolated = np.interp(unique_elapsed_time_values, trial['elapsed_time'], trial[col])
                        interpolated_values[col].append(interpolated)
            
            # Calculate averages for each column
            for col in columns_to_average:
                if interpolated_values[col]:  # Check if we have any values
                    average_trial[col] = np.mean(interpolated_values[col], axis=0)
            
            # Calculate averages for optional columns if they exist
            for col in optional_columns:
                if interpolated_values[col]:  # Check if we have any values
                    average_trial[col] = np.mean(interpolated_values[col], axis=0)
            
            # Store the average trial in the results dictionary
            if 'average_trajectories' not in results_dict['main']:
                results_dict['main']['average_trajectories'] = {}
            
            results_dict['main']['average_trajectories'][key] = average_trial
            
    return results_dict

    
def estimate_virtual_trajectories_single_participant(results_dict, use_corrected_outliers: bool = False): 
    """
    Estimate the virtual spring trajectories for a single participant. 
    To do this, we need their calibration results. We are going to set up some IVPs to estimate the virtual trajectory. 
    """
    
    inertia_slope = results_dict['calibration']['results']['linear_fits']['inertia']['slope']
    inertia_intercept = results_dict['calibration']['results']['linear_fits']['inertia']['intercept']
    damping_slope = results_dict['calibration']['results']['linear_fits']['damping']['slope']
    damping_intercept = results_dict['calibration']['results']['linear_fits']['damping']['intercept']
    stiffness_slope = results_dict['calibration']['results']['linear_fits']['stiffness']['slope']
    stiffness_intercept = results_dict['calibration']['results']['linear_fits']['stiffness']['intercept']
    
    # Define the phi_dot function for use with solve_ivp
    def phi_dot_ivp(t, y, theta, theta_dot, theta_ddot, grip_force, load):
        J = inertia_intercept + inertia_slope * grip_force
        b = damping_intercept + damping_slope * grip_force
        k = stiffness_intercept + stiffness_slope * grip_force
        
        # Use y[0] (phi) instead of theta in the first term
        return -(k/b) * y[0] + (1/b) * (k*theta + b*theta_dot + J*theta_ddot + load)
    
    for trial in results_dict['main']['runs']: 
        if trial.is_outlier() and not trial.is_corrected_outlier(): 
            continue 
        
        if use_corrected_outliers or not trial.is_outlier(): 
            # Extract data from the trial
            theta = trial['EncoderRadians_smooth'].to_numpy()
            theta_dot = trial['EncoderRadians_dot_smooth'].to_numpy()
            theta_ddot = trial['EncoderRadians_ddot_smooth'].to_numpy()
            grip_force = trial['GripForce'].to_numpy()
            load = np.clip(theta * trial.get_load() / 0.1, 0, trial.get_load())
            time = trial['elapsed_time'].to_numpy()
            
            # Create a wrapper function for solve_ivp that interpolates the trial data
            def phi_dot_wrapper(t, y):
                # Find the closest time index
                idx = np.argmin(np.abs(time - t))
                return phi_dot_ivp(t, y, theta[idx], theta_dot[idx], theta_ddot[idx], 
                                  grip_force[idx], load[idx])
            
            # Solve the IVP
            solution = solve_ivp(phi_dot_wrapper, [time[0], time[-1]], [0], 
                                t_eval=time, method='RK45')
            
            # Extract the solution
            virtual_trajectory = solution.y[0]
            
            # Calculate phi_dot for each time point
            virtual_trajectory_dot = np.array([phi_dot_wrapper(t, [phi]) for t, phi in zip(time, virtual_trajectory)])
            
            # Add both to the dataframe
            trial['VirtualTrajectory_Dot'] = virtual_trajectory_dot
            trial['VirtualTrajectory'] = virtual_trajectory
            
    return results_dict

def calculate_single_virtual_trajectory(results_dict: dict, df: pd.DataFrame, load_value: float): 
    """Calculate the virtual trajectory for a single trial. 
    """
    
    inertia_slope = results_dict['calibration']['results']['linear_fits']['inertia']['slope']
    inertia_intercept = results_dict['calibration']['results']['linear_fits']['inertia']['intercept']
    damping_slope = results_dict['calibration']['results']['linear_fits']['damping']['slope']
    damping_intercept = results_dict['calibration']['results']['linear_fits']['damping']['intercept']
    stiffness_slope = results_dict['calibration']['results']['linear_fits']['stiffness']['slope']
    stiffness_intercept = results_dict['calibration']['results']['linear_fits']['stiffness']['intercept']
    
    # Define the phi_dot function for use with solve_ivp
    def phi_dot_ivp(t, y, theta, theta_dot, theta_ddot, grip_force, load):
        J = inertia_intercept + inertia_slope * grip_force
        b = damping_intercept + damping_slope * grip_force
        k = stiffness_intercept + stiffness_slope * grip_force
        
        # Use y[0] (phi) instead of theta in the first term
        return -(k/b) * y[0] + (1/b) * (k*theta + b*theta_dot + J*theta_ddot + load)
    
    # Extract data from the trial
    theta = df['EncoderRadians_smooth'].to_numpy()
    theta_dot = df['EncoderRadians_dot_smooth'].to_numpy()
    theta_ddot = df['EncoderRadians_ddot_smooth'].to_numpy()
    grip_force = df['GripForce'].to_numpy()
    load = np.clip(theta * load_value / 0.1, 0, load_value)
    time = df['elapsed_time'].to_numpy()
    
    # Create a wrapper function for solve_ivp that interpolates the trial data
    def phi_dot_wrapper(t, y):
        # Find the closest time index
        idx = np.argmin(np.abs(time - t))
        return phi_dot_ivp(t, y, theta[idx], theta_dot[idx], theta_ddot[idx], 
                          grip_force[idx], load[idx])
        
    # Solve the IVP
    solution = solve_ivp(phi_dot_wrapper, [time[0], time[-1]], [0], 
                        t_eval=time, method='RK45')
    
    # Extract the solution
    virtual_trajectory = solution.y[0]
    virtual_trajectory_dot = np.array([phi_dot_wrapper(t, [phi]) for t, phi in zip(time, virtual_trajectory)])
    
    return virtual_trajectory, virtual_trajectory_dot
    
def estimate_scaling_factors_single_participant(results_dict): 
    """
    Estimate the scaling factors for a single participant. What this means is that we are going to form a few optimization problems. 
    We want to find the parameter alpha that best describes the relationship between the grip force during a light trial and the grip force during a medium trial. 
    We want to find the parameter beta that best describes the relationship between the grip force during a heavy trial and the grip force during a medium trial. 
    We want to find the parameter gamma that best describes the relationship between the virtual trajectory during a light trial and the virtual trajectory during a medium trial
    We want to find the parameter eta that best describes the relationship between the virtual trajectory during a heavy trial and the virtual trajectory during a medium trial. 
    
    For heavy/light trials, we are going to use the averages. 
    """
    
    light_trajectory = results_dict['main']['average_trajectories']['0.1_normal_visible']
    heavy_trajectory = results_dict['main']['average_trajectories']['0.5_normal_visible']
    medium_trajectory = results_dict['main']['average_trajectories']['0.3_normal_visible']
    
    
    
    
    
    
    
            
                    

### NEW - MAR 28 2025
def extract_individual_runs(filepath : str, trial_type: str) -> List:
    
    # First get the individual trials
    df = read_and_preprocess_data(filepath)
    live_indices = find_live_indices(df)
    
    trials = []
    for index in live_indices: 
        start, end = index 
        # Modify the end index to be the index closest to 0.2 seconds in elapsed time. 
        trial = df.iloc[start:end].copy()
        trial['elapsed_time'] = trial['time_s'] - trial['time_s'].min()
        trial = estimate_velocity_and_accel(trial)
        
        if trial_type == 'calibration': 
            trials.append(CalibrationTrial(trial))
        elif trial_type == 'main': 
            trials.append(MainTrial(trial))
        
    return trials

def detect_calibration_outliers(trials : List[CalibrationTrial]) -> List[CalibrationTrial]: 
    """
    Detect outliers in the calibration trials. 
    """
    for trial in trials: 
        # If the steady state position value is outside of absolute value 0.6 radians, 
        # then we consider it an outlier. 
        if np.abs(trial['EncoderRadians_smooth'].iloc[-1]) > 0.6: 
            trial.mark_outlier()
        
    return trials

def detect_main_outliers(trials: List[MainTrial]) -> int: 
    """
    Detect outliers in the main trials. Main criteria is the trial length. However,
    some trials can be corrected for by adjusting the start index if this occurs.     
    Unlike calibration trials, we do not return the trials since the objects will be marked in place. 
    """
    num_outliers = 0 
    num_corrected_outliers = 0 
    for trial in trials: 
        if trial['elapsed_time'].max() > 3.5 and not trial.is_catch_trial(): 
            trial.mark_outlier()
            num_outliers += 1
        
        # If past 0.15 seconds in elapsed time, the encoder position is not at least 0.5, then it is an outlier. 
        elif trial['EncoderRadians_smooth'].iloc[int(0.15 / trial['elapsed_time'].iloc[1])] < 0.5: 
            trial.mark_outlier()
            
            # Try to correct for the outlier by adjusting the start index. Find the first time where the 
            # position goes above 0.025 radians and make that the new start index. 
            start_index = np.where(trial['EncoderRadians_smooth'] > 0.025)[0][0]
            trial.correct_start_index(start_index)
            trial.correct_zero_position() 
            trial.mark_corrected_outlier()
            num_corrected_outliers += 1
            num_outliers += 1
        else: 
            continue 
            
    return trials, num_outliers, num_corrected_outliers
    
def calculate_complete_calibration_metrics(trials: List[CalibrationTrial]) -> dict: 
    """
    Calculate the m/b/k estimates for the calibration trials. 
    Uses all trials with the same grip force threshold to calculate one estimate. 
    """
    trials_by_grip_force_threshold = {} 
    
    for trial in trials: 
        grip_threshold = trial['GripThreshold'].iloc[0]  # Get scalar value from Series
        if grip_threshold not in trials_by_grip_force_threshold:
            trials_by_grip_force_threshold[grip_threshold] = []
        trials_by_grip_force_threshold[grip_threshold].append(trial)
        
    results = {}
    results['by_grip_force_threshold'] = {}
    results['linear_fits'] = {}
    for grip_force_threshold, trial_list in trials_by_grip_force_threshold.items(): 
        results['by_grip_force_threshold'][grip_force_threshold] = calculate_calibration_metrics_for_batch(trial_list)
    
    # print(f"Completed {len(trials)} trials. Here are the calibration results:")    
    # for threshold, metrics in results.items():
    #     print(f"\nGrip Force Threshold: {threshold}")
    #     print(f"  Inertia: {float(metrics['inertia'][0]):.4f}")
    #     print(f"  Damping: {float(metrics['damping'][0]):.4f}")
    #     print(f"  Stiffness: {float(metrics['stiffness'][0]):.4f}")
    #     print(f"  Residual Error: {float(metrics['residual_error']):.4f}")
    #     print(f"  Number of Valid Trials: {metrics['num_valid_trials']}")
    #     print(f"  Number of Total Trials: {metrics['num_total_trials']}")
        
    # Calculate linear fits for inertia, damping, and stiffness vs grip force threshold
    grip_thresholds = np.array(list(results['by_grip_force_threshold'].keys()))
    
    # Extract metrics into arrays
    inertias = np.array([res['inertia'] for res in results['by_grip_force_threshold'].values()])
    dampings = np.array([res['damping'] for res in results['by_grip_force_threshold'].values()])
    stiffnesses = np.array([res['stiffness'] for res in results['by_grip_force_threshold'].values()])
    
    # Calculate linear fits
    inertia_fit = np.polyfit(grip_thresholds, inertias, 1)
    damping_fit = np.polyfit(grip_thresholds, dampings, 1)
    stiffness_fit = np.polyfit(grip_thresholds, stiffnesses, 1)
    
    # Add fits to results dict
    results['linear_fits'] = {
        'inertia': {'slope': inertia_fit[0], 'intercept': inertia_fit[1]},
        'damping': {'slope': damping_fit[0], 'intercept': damping_fit[1]},
        'stiffness': {'slope': stiffness_fit[0], 'intercept': stiffness_fit[1]}
    }
        
    return results 

def calculate_calibration_metrics_for_batch(trials: List[CalibrationTrial], max_time : float = 0.1) -> dict: 
    """
    Calculate the m/b/k estimates for a batch of trials.  
    All trials passed in must have the same grip force threshold. 
    Outliers will be removed from the calculation. 
    """
    results = {} 
    
    # We first need to collect all the trials into a single set of arrays.
    # Only include non-outlier trials
    valid_trials = [trial for trial in trials if not trial.is_outlier()]
    
    # Concatenate the data sequences vertically into column vectors
    thetas = np.concatenate([trial['EncoderRadians_smooth'][trial['elapsed_time'] <= max_time].values for trial in valid_trials]).reshape(-1, 1)
    thetas_dot = np.concatenate([trial['EncoderRadians_dot_smooth'][trial['elapsed_time'] <= max_time].values for trial in valid_trials]).reshape(-1, 1) 
    thetas_ddot = np.concatenate([trial['EncoderRadians_ddot_smooth'][trial['elapsed_time'] <= max_time].values for trial in valid_trials]).reshape(-1, 1)
    torques = np.concatenate([trial['CommandTorque'][trial['elapsed_time'] <= max_time].values for trial in valid_trials]).reshape(-1, 1)
    
    # Now form design matrices
    A = np.column_stack((thetas_ddot, thetas_dot, thetas))
    b = torques 
    
    # Now solve for x, the vector of m/b/k estimates. 
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    results['inertia'] = x[0][0]
    results['damping'] = x[1][0]
    results['stiffness'] = x[2][0]
    results['residual_error'] = residuals[0]
    results['num_valid_trials'] = len(valid_trials)
    results['num_total_trials'] = len(trials)
    
    return results
    
    

def process_individual_calibration_run(filepath : str) -> dict: 

    # First get the individual trials
    df = read_and_preprocess_data(filepath)
    live_indices = find_live_indices(df) 
    
    trials = {} 
    
    for i, index in enumerate(live_indices): 
        
        # Create a dict for the trial index 
        trials[i] = {}
        
        start, end = index
        end = start + 700 # 200ms of data, TODO: look into if this should be changed. 
        trial = df.iloc[start:end].copy()
        trial['elapsed_time'] = trial['time_s'] - trial['time_s'].min()
        trial = estimate_velocity_and_accel(trial) # Estimate only in trial window. 
        trials[i]['data'] = trial 

        # Now we weed out individual trials based on some heuristics. 
        trials[i]['is_valid'] = check_calibration_trial_validity(trial)

        # If the trial is valid, calculate the calibration metrics 
        if trials[i]['is_valid']: 
            trials[i]['metrics'] = calculate_calibration_metrics(trial)
            # If residual error is greater than 40, discard the trial. 
            if trials[i]['metrics']['residual_error'] > 100: 
                trials[i]['is_valid'] = False 


    # Then we aggregate the results into a "metrics" field 
    metrics_df = pd.DataFrame(columns=['Trial Number', 'Grip Force Threshold', 'Inertia', 'Damping', 'Stiffness', 'Residual Error'])
    for i in range(len(trials)): 
        if trials[i]['is_valid']: 
            metrics_df.loc[len(metrics_df)] = {
                'Trial Number': i,
                'Grip Force Threshold': trials[i]['metrics']['grip_force_threshold'],
                'Inertia': trials[i]['metrics']['inertia'], 
                'Damping': trials[i]['metrics']['damping'],
                'Stiffness': trials[i]['metrics']['stiffness'],
                'Residual Error': trials[i]['metrics']['residual_error']
            }

    results = {"trials": trials, "num_trials": len(trials), "metrics": metrics_df}
    return results

def check_calibration_trial_validity(trial : pd.DataFrame) -> bool: 

    # Apply some heuristics to check if the trial is valid. 
    # For now, if the grip force ever returns to its initial value after the first 
    # 10 data points, we consider it invalid 
    grip_force_dot = trial['GripForce_dot_smooth']
    neg_velocity_threshold = -1000
    if np.any(grip_force_dot < neg_velocity_threshold): 
        return False 
    
    return True 

def process_individual_main_run(filepath : str) -> dict: 

    # First get the individual trials 
    df = pd.read_csv(filepath) 
    df.columns = df.columns.str.strip()
    data = add_time_columns(df) 
    live_indices = find_live_indices(df) 
    
    trials = {} 
    for i, index in enumerate(live_indices): 
        # Create a dict for the trial index
        trials[i] = {}
        
        start, end = index 

        # First get trial with original indices and estimate velocity/accel
        trial = df.iloc[start:end].copy()
        trial['elapsed_time'] = trial['time_s'] - trial['time_s'].min()
        trial = estimate_velocity_and_accel(trial)
        
        # Check if we need to adjust start index based on velocity threshold
        velocity = trial['EncoderRadians_dot_smooth']
        if np.any(velocity > 0):
            # Get new start index and reprocess trial
            new_start = start + np.where(velocity > 0)[0][0]
            trial = df.iloc[new_start:end].copy()
            trial['elapsed_time'] = trial['time_s'] - trial['time_s'].min()
            trial = estimate_velocity_and_accel(trial)
        trials[i]['data'] = trial
        trials[i]['is_valid'] = check_main_trial_validity(trial)
        trials[i]['load'] = load_map[np.max(np.abs(trial['CommandTorque']))]
        trials[i]['is_catch'] = True if i != 0 and trials[i-1]['load'] != trials[i]['load'] else False
        
    results = {"trials": trials, "num_trials": len(trials)}
    return results

def check_main_trial_validity(trial : pd.DataFrame) -> bool: 
    # # For now, see if the trial is shorter than 2 seconds. 
    # if trial['elapsed_time'].max() >2: 
    #     return False 

    # # Check if the angle is not at least 0.5 radians by 0.3 seconds into the trial 
    time_gate_index = int(0.2 / trial['elapsed_time'].iloc[1])
    if trial['EncoderRadians_smooth'].iloc[time_gate_index] < 0.5: 
        return False 
    
    return True 
    

def read_and_preprocess_data(filename : str) -> pd.DataFrame: 
    data = pd.read_csv(filename) 
    
    # Remove any whitespace if applicable 
    data.columns = data.columns.str.strip() 
    
    # Rename some columns: 
    data.columns = data.columns.str.lower()
    data = data.rename(columns={
        'ntimestephigh': 'TimeHigh',
        'ntimesteplow': 'TimeLow',
        'ntestislive': 'TestIsLive',
        'rencoderdegrees': 'EncoderDegrees',
        'rencoderradians': 'EncoderRadians', 
        'rcommandvoltage': 'CommandVoltage',
        'rcommandtorque': 'CommandTorque',
        'rappliedtorque': 'AppliedTorque',
        'rwalltorque': 'WallTorque',
        'rvirtualspringtorque': 'VirtualSpringTorque',
        'rgripforce': 'GripForce',
        'rcurrentgripthreshold': 'GripThreshold',
        'ssequencestatestring': 'SequenceStateString',
        'ncurrenttestindex': 'CurrentTestIndex',
    })
    
    # Truncate torque columns to 5 decimal places if they exist
    torque_columns = ['CommandTorque', 'AppliedTorque', 'WallTorque', 'VirtualSpringTorque']
    for col in torque_columns:
        if col in data.columns:
            data[col] = data[col].round(5)
    
    data = add_time_columns(data)
    # data = estimate_velocity_and_accel(data)
    
    return data 
    
def add_time_columns(data : pd.DataFrame) -> pd.DataFrame: 
    """
    Add time columns to the data. Converts Beckhoff's time columns 
    into nanoseconds, milliseconds, and seconds. 
    """
    
    # Assure this doesn't run twice. 
    if 'time_ns' in data.columns: 
        return data 
    
    # Before processing, we have a 'TimeHigh' and 'TimeLow' column. 
    # which is the binary representation of the time in 100s of ns 
    data['time_ns'] = (data['TimeHigh'].astype('int64').values << 32 | data['TimeLow'].astype('int64').values).astype('float64') * 100
    data['time_ms'] = data['time_ns'] / 1e6
    data['time_s'] = data['time_ns'] / 1e9

    # Then fix columns so they start at 0
    data['time_ns'] = data['time_ns'] - data['time_ns'].min()
    data['time_ms'] = data['time_ms'] - data['time_ms'].min()
    data['time_s'] = data['time_s'] - data['time_s'].min()
    
    return data 


def estimate_velocity_and_accel(data : pd.DataFrame) -> pd.DataFrame: 
    """
    Estimate velocity and acceleration from the position data. 
    Uses a Savitzky-Golay filter to smooth the data, and provides
    both smoothed and unsmoothed estimates of velocity and acceleration. 

    Also, estimate the grip force velocity. 
    """
    
    polyorder = 3
    window_length = 101 
    
    data['EncoderRadians'] = data['EncoderRadians'] - data['EncoderRadians'].iloc[0] # TESTING: Remove offset. 
    data['EncoderRadians_smooth'] = savgol_filter(data['EncoderRadians'], window_length, polyorder)
    data['EncoderRadians_dot'] = np.gradient(data['EncoderRadians_smooth'], data['time_s'])
    data['EncoderRadians_dot_smooth'] = savgol_filter(data['EncoderRadians_dot'], window_length, polyorder)
    data['EncoderRadians_ddot'] = np.gradient(data['EncoderRadians_dot_smooth'], data['time_s'])
    data['EncoderRadians_ddot_smooth'] = savgol_filter(data['EncoderRadians_ddot'], window_length, polyorder)

    # Estimate the grip force velocity 
    data['GripForce_dot'] = np.gradient(data['GripForce'], data['time_s'])
    data['GripForce_dot_smooth'] = savgol_filter(data['GripForce_dot'], window_length, polyorder)
    
    return data

    
def find_live_indices(data : pd.DataFrame) -> list[tuple[int, int]]: 
    live_indices = []
    start = None 
    for i, row in data.iterrows(): 
        if row['TestIsLive'] and start is None: 
            start = i 
        elif not row['TestIsLive'] and start is not None: 
            live_indices.append((start, i))
            start = None 
            
    if start is not None: 
        live_indices.append((start, len(data) - 1))
        
    return live_indices

    
def calculate_calibration_metrics(trial : pd.DataFrame) -> dict: 
    """
    Calculate the calibration metrics for a single trial. 

    Return a dictionary with the following keys: 
    - 'inertia' : float
    - 'damping' : float
    - 'stiffness' : float
    - 'residual_error' : float
    - 'grip_force_threshold' : float
    """    
     
    results = {} 
    
    thetas = trial['EncoderRadians_smooth']
    thetas_dot = trial['EncoderRadians_dot_smooth']
    thetas_ddot = trial['EncoderRadians_ddot_smooth']
    torques = trial['CommandTorque']
    
    # Now form matrices
    A = np.column_stack((thetas_ddot, thetas_dot, thetas))
    b = torques 
    
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    results['inertia'] = x[0]
    results['damping'] = x[1]
    results['stiffness'] = x[2]
    results['residual_error'] = residuals[0]
    results['grip_force_threshold'] = trial['GripThreshold'].iloc[0]
    
    return results 
     