import matplotlib.pyplot as plt 
import pandas as pd 
import os 
import numpy as np 
from processing import split_runs_into_trial_types
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from functools import partial
from TrialData import MainTrial, CalibrationTrial
def plot_trial_list(trial_list, title=None, show_outliers=False, just_outliers=False, save_fig=False, save_path=None):
    """
    Plot multiple trials overlaid on the same subplots.
    Each trial dataframe contains elapsed_time and sensor data columns.
    When show_outliers=True, outliers are red and valid trials are gray.
    When show_outliers=False, each trial gets a unique color.
    When just_outliers=True, only outlier trials are shown.
    
    Args:
        trial_list: List of trial dataframes to plot
        title: Optional title for the figure
        show_outliers: Whether to highlight outliers in red
        just_outliers: Whether to only show outlier trials
        save_fig: Whether to save the figure
        save_path: Path to save the figure if save_fig is True
    """
    # Create figure and subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    
    if title:
        fig.suptitle(title)
    
    for i, trial_df in enumerate(trial_list):
        # Skip non-outliers when just_outliers is True
        if just_outliers and not trial_df.is_outlier():
            continue
            
        # Determine color and alpha based on outlier status and show_outliers flag
        if show_outliers:
            if trial_df.is_outlier() and not just_outliers:
                if trial_df.is_corrected_outlier():
                    color = 'green'
                else: 
                    color = 'red'
                alpha = 0.7
            elif trial_df.is_outlier() and just_outliers:
                if trial_df.is_corrected_outlier():
                    color = 'green'
                else: 
                    color = 'red'
                alpha = 0.5
            else:
                color = 'gray'
                alpha = 0.5
        else:
            # Use color cycle when not showing outliers
            color = None  # Will use next color from matplotlib's color cycle
            alpha = 0.5
            
        # Plot position data
        ax1.plot(trial_df['elapsed_time'], trial_df['EncoderRadians_smooth'], 
                color=color, alpha=alpha)
        
        # Plot velocity data
        ax2.plot(trial_df['elapsed_time'], trial_df['EncoderRadians_dot_smooth'], 
                color=color, alpha=alpha)
        
        # Plot acceleration data
        ax3.plot(trial_df['elapsed_time'], trial_df['EncoderRadians_ddot_smooth'], 
                color=color, alpha=alpha)
        
        # Plot torque data
        if 'AppliedTorque' in trial_df:
            ax4.plot(trial_df['elapsed_time'], trial_df['AppliedTorque'], 
                    color=color, alpha=alpha)
        else:
            ax4.plot(trial_df['elapsed_time'], trial_df['WallTorque'], 
                    color=color, alpha=alpha)
            
        # Plot grip force data
        ax5.plot(trial_df['elapsed_time'], trial_df['GripForce'], 
                color=color, alpha=alpha)

    # Set labels and titles
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Encoder Position')
    ax1.grid(True)

    ax2.set_ylabel('Velocity (rad/s)') 
    ax2.set_title('Encoder Velocity')
    ax2.grid(True)

    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Encoder Acceleration')
    ax3.grid(True)

    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Applied Torque')
    ax4.grid(True)

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Force (N)')
    ax5.set_title('Grip Force')
    ax5.grid(True)

    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_path:
        plt.savefig(save_path)
        
    return fig, (ax1, ax2, ax3, ax4, ax5)


def plot_calibration_results_for_batch_and_individual(calibration_results, trial_list, title=None, save_fig=False, save_path=None): 
    """
    Plot the calibration results for a batch of trials and individual trials.

    Args:
        calibration_results (dict): Dictionary containing batch calibration results for each grip force threshold
        trial_list (list): List of CalibrationTrial objects to plot individual results for
        title (str, optional): Title for the figure
        save_fig (bool, optional): Whether to save the figure
        save_path (str, optional): Path to save the figure if save_fig is True

    Returns:
        fig: The matplotlib figure object
        axes: The matplotlib axes objects
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    if title:
        fig.suptitle(title)
    
    # Plot batch results
    grip_thresholds = np.array(list(calibration_results['by_grip_force_threshold'].keys()))
    inertias = np.array([res['inertia'] for res in calibration_results['by_grip_force_threshold'].values()])
    dampings = np.array([res['damping'] for res in calibration_results['by_grip_force_threshold'].values()])
    stiffnesses = np.array([res['stiffness'] for res in calibration_results['by_grip_force_threshold'].values()])
    
    ax1.plot(grip_thresholds, inertias, 'kx', markersize=10, label='Batch Fit')
    ax2.plot(grip_thresholds, dampings, 'kx', markersize=10, label='Batch Fit') 
    ax3.plot(grip_thresholds, stiffnesses, 'kx', markersize=10, label='Batch Fit')
    
    # Plot linear fits
    ax1.plot(grip_thresholds, calibration_results['linear_fits']['inertia']['slope'] * grip_thresholds + calibration_results['linear_fits']['inertia']['intercept'], 'r-', label='Linear Fit')
    ax2.plot(grip_thresholds, calibration_results['linear_fits']['damping']['slope'] * grip_thresholds + calibration_results['linear_fits']['damping']['intercept'], 'r-', label='Linear Fit')
    ax3.plot(grip_thresholds, calibration_results['linear_fits']['stiffness']['slope'] * grip_thresholds + calibration_results['linear_fits']['stiffness']['intercept'], 'r-', label='Linear Fit')
    
    
    # Plot individual trial results
    for trial in trial_list:
        if not trial.is_outlier():
            metrics = trial.calculate_metrics()
            grip_threshold = trial['GripThreshold'].iloc[0]
            
            ax1.scatter(grip_threshold, metrics['inertia'], c='b', marker='o', alpha=0.3)
            ax2.scatter(grip_threshold, metrics['damping'], c='b', marker='o', alpha=0.3)
            ax3.scatter(grip_threshold, metrics['stiffness'], c='b', marker='o', alpha=0.3)

    # Labels and formatting
    ax1.set_ylabel('Inertia (kg⋅m²)')
    ax1.set_title('Inertia vs Grip Force Threshold')
    ax1.grid(True)
    ax1.legend()

    ax2.set_ylabel('Damping (N⋅m⋅s/rad)')
    ax2.set_title('Damping vs Grip Force Threshold') 
    ax2.grid(True)

    ax3.set_xlabel('Grip Force Threshold (N)')
    ax3.set_ylabel('Stiffness (N⋅m/rad)')
    ax3.set_title('Stiffness vs Grip Force Threshold')
    ax3.grid(True)

    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_path:
        plt.savefig(save_path)
        
    return fig, (ax1, ax2, ax3)


def plot_main_load_schedule(trial_list : list[MainTrial], title=None, save_fig=False, save_path=None): 
    """
    Plot the load schedule for a list of MainTrial objects.
    
    Creates two subplots:
    1. Bar graph showing:
       - X-axis: Trial number
       - Y-axis: Maximum applied torque (Nm)
       - Bar colors: Indicate whether loads were visible or hidden
    2. Line plot showing trial durations
       - Outlier trials are marked with a red X
    
    Parameters:
        trial_list: List of MainTrial objects to plot
        title: Optional title for the figure
        save_fig: Whether to save the figure
        save_path: Path to save the figure if save_fig is True
        
    Returns:
        fig, axes: The matplotlib figure and axes objects
    """
    # Create figure with subplots - main plot for torques and smaller plot for durations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], gridspec_kw={'hspace': 0.3})
    
    if title:
        fig.suptitle(title)
    
    # Prepare data
    trial_numbers = list(range(1, len(trial_list) + 1))
    max_torques = [trial['AppliedTorque'].max() if 'AppliedTorque' in trial else trial['WallTorque'].max() 
                  for trial in trial_list]
    durations = [trial['elapsed_time'].max() for trial in trial_list]
    visibility = [trial.is_load_visible() for trial in trial_list]
    is_outlier = [trial.is_outlier() for trial in trial_list]
    
    # Define colors for visibility
    colors = ['#3498db' if vis else '#e74c3c' for vis in visibility]
    
    # Create bar plot for torques
    ax1.bar(trial_numbers, max_torques, color=colors, alpha=0.7)
    
    # Add legend for visibility
    visible_patch = plt.Rectangle((0, 0), 1, 1, color='#3498db', alpha=0.7, label='Loads Shown')
    hidden_patch = plt.Rectangle((0, 0), 1, 1, color='#e74c3c', alpha=0.7, label='Loads Hidden')
    ax1.legend(handles=[visible_patch, hidden_patch], loc='upper right')
    
    # Set axis properties for torque plot
    ax1.set_ylabel('Maximum Applied Torque (Nm)')
    ax1.set_title('Trial Load Schedule')
    
    # Only show every 5th tick to avoid overcrowding
    show_ticks = [i for i in trial_numbers if i % 5 == 0 or i == 1 or i == len(trial_numbers)]
    ax1.set_xticks(show_ticks)
    ax1.set_xticklabels([f'#{i}' for i in show_ticks])
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Create line plot for durations
    # Split data into normal and outlier trials
    normal_trials = [i for i, outlier in enumerate(is_outlier) if not outlier]
    outlier_trials = [i for i, outlier in enumerate(is_outlier) if outlier]
    corrected_outlier_trials = [i for i, outlier in enumerate(is_outlier) if outlier and trial_list[i].is_corrected_outlier()]
    
    # Plot normal trials with line and circles
    if normal_trials:
        normal_x = [trial_numbers[i] for i in normal_trials]
        normal_y = [durations[i] for i in normal_trials]
        ax2.plot(normal_x, normal_y, 'o-', color='#2ecc71', linewidth=2, markersize=5, label='Normal Trials')
    
    # Plot outlier trials with red X markers
    if outlier_trials:
        outlier_x = [trial_numbers[i] for i in outlier_trials]
        outlier_y = [durations[i] for i in outlier_trials]
        ax2.plot(outlier_x, outlier_y, 'rx', markersize=8, markeredgewidth=2, label='Outlier Trials')
        
    if corrected_outlier_trials: 
        corrected_outlier_x = [trial_numbers[i] for i in corrected_outlier_trials]
        corrected_outlier_y = [durations[i] for i in corrected_outlier_trials]
        ax2.plot(corrected_outlier_x, corrected_outlier_y, 'gx', markersize=8, markeredgewidth=2, label='Corrected Outlier Trials')
    
    # Add legend for duration plot
    if outlier_trials:
        ax2.legend(loc='upper right')
        
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Duration (s)')
    ax2.set_title('Trial Durations')
    ax2.set_xticks(show_ticks)
    ax2.set_xticklabels([f'#{i}' for i in show_ticks])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_path:
        plt.savefig(save_path)
        
    return fig, (ax1, ax2)
    

def plot_single_trial(trial):
    """
    Plot a single trial with position, velocity, acceleration, torque, and grip force.
    
    Parameters:
        trial: A MainTrial or CalibrationTrial object containing the trial data
        
    Returns:
        fig: The matplotlib figure object
    """
    print(f"Plotting trial")
    # Create figure and subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    
    # Get data from trial object
    elapsed_time = trial['elapsed_time']
    
    # Plot position data
    ax1.plot(elapsed_time, trial['EncoderRadians_smooth'], 'b-', label='Position')
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Encoder Position')
    ax1.grid(True)
    ax1.legend()
    
    # Plot velocity data  
    ax2.plot(elapsed_time, trial['EncoderRadians_dot_smooth'], 'g-', label='Velocity')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Encoder Velocity')
    ax2.grid(True)
    ax2.legend()
    
    # Plot acceleration data
    ax3.plot(elapsed_time, trial['EncoderRadians_ddot_smooth'], 'r-', label='Acceleration')
    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Encoder Acceleration')
    ax3.grid(True)
    ax3.legend()
    
    # Plot torque data
    if 'AppliedTorque' in trial: 
        ax4.plot(elapsed_time, trial['AppliedTorque'], 'k-', label='Torque')
    else: 
        ax4.plot(elapsed_time, trial['Command Torque'], 'k-', label='Torque')
    
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Command Torque')
    ax4.grid(True)
    ax4.legend()
    
    # Plot grip force data
    ax5.plot(elapsed_time, trial['GripForce'], 'm-', label='Grip Force')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Force (N)')
    ax5.set_title('Grip Force')
    ax5.grid(True)
    ax5.legend()
    
    # Set x-axis ticks at 0.1 second intervals
    trial_duration = elapsed_time.iloc[-1]
    xticks = np.arange(0, trial_duration + 0.1, 0.1)
    ax5.set_xticks(xticks)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig


def plot_average_trajectories_single_participant(results_dict): 
    """
    Plot the average trajectories for a single participant. 
    """
    
    # Create a 3x2 grid where the last position will be used for the legend
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    legend_ax = fig.add_subplot(gs[2, 1])
    
    # Store line objects for the legend
    lines = []
    labels = []
    
    for key, trial in results_dict['main']['average_trajectories'].items(): 
        # Parse the key components
        parts = key.split('_')
        load_value = parts[0]
        trial_type = parts[1]  # 'normal' or 'catch'
        visibility = parts[2]  # 'visible' or 'hidden'
        
        # Set line style based on trial type
        linestyle = 'dotted' if trial_type == 'catch' else 'solid'
        
        # Set marker based on visibility
        marker = 'o' if visibility == 'visible' else 'x'
        
        # We'll use markevery to avoid cluttering the plot with too many markers
        markevery = 10
        
        # Create a custom line for each trial with appropriate styling
        line1, = ax1.plot(trial['elapsed_time'], trial['EncoderRadians_smooth'], linestyle=linestyle, marker=marker, markevery=markevery)
        ax2.plot(trial['elapsed_time'], trial['EncoderRadians_dot_smooth'], color=line1.get_color(), linestyle=linestyle, marker=marker, markevery=markevery)
        ax3.plot(trial['elapsed_time'], trial['EncoderRadians_ddot_smooth'], color=line1.get_color(), linestyle=linestyle, marker=marker, markevery=markevery)
        ax4.plot(trial['elapsed_time'], trial['CommandTorque'], color=line1.get_color(), linestyle=linestyle, marker=marker, markevery=markevery)
        ax5.plot(trial['elapsed_time'], trial['GripForce'], color=line1.get_color(), linestyle=linestyle, marker=marker, markevery=markevery)
        
        # Store the line and label for the legend
        lines.append(line1)
        labels.append(key)
    
    # Add axis labels
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Encoder Position')
    ax1.grid(True)
    
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Encoder Velocity')
    ax2.grid(True)
    
    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Encoder Acceleration')
    ax3.grid(True)
    
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Command Torque')
    ax4.grid(True)
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Force (N)')
    ax5.set_title('Grip Force')
    ax5.grid(True)
    
    # Limit x-axis to 0.5 seconds
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlim(0, 0.5)
    
    # Create the legend in the empty subplot
    legend_ax.axis('off')
    legend_ax.legend(lines, labels, loc='center', fontsize='large')
    
    plt.tight_layout()
    
    return fig


def plot_average_trajectories_single_participant_for_paper(results_dict, title=None, save_fig=False, save_path=None): 
    """
    Plot the average trajectories for a single participant, focusing on position and grip force
    for specific load conditions.
    
    Creates a 2x3 grid of subplots showing:
    1. Heavy load (0.5): catch vs. non-catch position
    2. Light load (0.1): catch vs. non-catch position
    3. Medium load (0.3): first catch vs. other catch (hidden) position
    4. Heavy load (0.5): catch vs. non-catch grip force
    5. Light load (0.1): catch vs. non-catch grip force
    6. Medium load (0.3): first catch vs. other catch (hidden) grip force
    
    Args:
        results_dict: Dictionary containing the results for a single participant
        title: Optional title for the figure
        save_fig: Whether to save the figure
        save_path: Path to save the figure if save_fig is True
        
    Returns:
        fig: The matplotlib figure object
    """
    
    # Create a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if title:
        fig.suptitle(title)
    
    # Define the specific cases we want to plot
    cases = {
        'heavy': {
            'Heavy Normal': '0.5_normal_hidden',
            'Heavy Catch': '0.5_catch_hidden'
        },
        'light': {
            'Light Normal': '0.1_normal_hidden',
            'Light Catch': '0.1_catch_hidden'
        },
        'medium': {
            'First Medium Visible': 'first_medium_catch_visible',
            'Avg Medium Catch': '0.3_catch_hidden'
        }
    }
    
    # Define line styles and colors
    styles = {
        'Heavy Normal': {'linestyle': 'solid', 'color': 'blue', 'label': 'Heavy Normal'},
        'Heavy Catch': {'linestyle': 'dotted', 'color': 'red', 'label': 'Heavy Catch'},
        'Light Normal': {'linestyle': 'solid', 'color': 'green', 'label': 'Light Normal'},
        'Light Catch': {'linestyle': 'dotted', 'color': 'purple', 'label': 'Light Catch'},
        'First Medium Visible': {'linestyle': 'solid', 'color': 'orange', 'label': 'First Medium Visible'},
        'Avg Medium Catch': {'linestyle': 'dotted', 'color': 'brown', 'label': 'Avg Medium Catch'}
    }
    
    # Plot position data (top row)
    for col, load_type in enumerate(['heavy', 'light', 'medium']):
        ax = axes[0, col]
        
        for trial_type, key in cases[load_type].items():
            if key in results_dict['main']['average_trajectories']:
                trial = results_dict['main']['average_trajectories'][key]
                ax.plot(trial['elapsed_time'], trial['EncoderRadians_smooth'], 
                       color=styles[trial_type]['color'],
                       linestyle=styles[trial_type]['linestyle'],
                       label=styles[trial_type]['label'])
        
        ax.set_xlim(0, 0.5)
        ax.set_ylabel('Position (rad)' if col == 0 else '')
        ax.set_title(f'{load_type.capitalize()} Load Position')
        ax.grid(True)
        ax.legend()
    
    # Plot grip force data (bottom row)
    for col, load_type in enumerate(['heavy', 'light', 'medium']):
        ax = axes[1, col]
        
        for trial_type, key in cases[load_type].items():
            if key in results_dict['main']['average_trajectories']:
                trial = results_dict['main']['average_trajectories'][key]
                ax.plot(trial['elapsed_time'], trial['GripForce'], 
                       color=styles[trial_type]['color'],
                       linestyle=styles[trial_type]['linestyle'],
                       label=styles[trial_type]['label'])
        
        ax.set_xlim(0, 0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Grip Force (N)' if col == 0 else '')
        ax.set_title(f'{load_type.capitalize()} Load Grip Force')
        ax.grid(True)
        ax.legend()
        
    # Add semi-transparent green region to show target area (60 degrees +/- 8 degrees)
    # Convert degrees to radians: 60 degrees = π/3 radians, 8 degrees = π/22.5 radians
    target_center = np.deg2rad(60)  # 60 degrees in radians
    target_range = np.deg2rad(8)  # 8 degrees in radians
    
    # Add the target region to each position plot in the top row
    for col in range(3):
        ax = axes[0, col]
        # Add semi-transparent green rectangle spanning the full x-axis range
        # with y-axis range from (target_center - target_range) to (target_center + target_range)
        ax.axhspan(
            target_center - target_range, 
            target_center + target_range, 
            color='green', 
            alpha=0.2, 
            label='Target Region'
        )
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig and save_path:
        plt.savefig(save_path)
    
    return fig

    
def plot_learning_curves(results_dict, title=None, save_fig=False, save_path=None): 
    """
    Plot the learning curves for a single participant. Generate two scatter plots, one for heavy load, one for light load. 
    On the scatter plot, we want 0-90 rise time v.s. trial number. 
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    
    # First calculate the 0-90 rise time for the first 10 trials for each load condition
    light_load_0_90_rise_times = []
    heavy_load_0_90_rise_times = []
    
    for trial in results_dict['main']['runs']: 
        if len(light_load_0_90_rise_times) >= 10 and len(heavy_load_0_90_rise_times) >= 10: 
            break
        
        if trial.is_load_visible(): 
            position_over_threshold = trial[trial['EncoderRadians_smooth'] > 0.9]
            if not position_over_threshold.empty:
                rise_time = position_over_threshold['elapsed_time'].iloc[0]
            else:
                rise_time = np.nan
            
            if trial.get_load() == 0.1: 
                light_load_0_90_rise_times.append(rise_time)
            elif trial.get_load() == 0.5: 
                heavy_load_0_90_rise_times.append(rise_time)
    
    # Now plot the data. Want rise times on y axis, trial number on x axis
    ax1.scatter(range(len(light_load_0_90_rise_times)), light_load_0_90_rise_times)
    ax2.scatter(range(len(heavy_load_0_90_rise_times)), heavy_load_0_90_rise_times)
    
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('0-90 Rise Time (s)')
    ax1.set_title('Light Load (0.1)')
    ax1.grid(True)
    
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('0-90 Rise Time (s)')
    ax2.set_title('Heavy Load (0.5)')
    ax2.grid(True)
    
    if title: 
        fig.suptitle(title)
    
    if save_fig and save_path: 
        plt.savefig(save_path)
    
    return fig


def plot_learning_curves_group(results_dict, title=None, save_fig=False, save_path=None): 
    """
    Plot the learning curves for a group of participants. Generate two scatter plots, one for heavy load, one for light load. 
    On the scatter plot, we want 0-90 rise time v.s. trial number for each participant with different colors.
    
    Args:
        results_dict (dict): Dictionary with participant IDs as keys and their results as values
        title (str, optional): Title for the plot
        save_fig (bool, optional): Whether to save the figure
        save_path (str, optional): Path to save the figure
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14))
    
    # Create a colormap for different participants
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_dict)))
    
    # Lists to collect all data points for exponential fitting
    all_light_trials = []
    all_light_times = []
    all_heavy_trials = []
    all_heavy_times = []
    
    # Process each participant's data
    for i, (participant, participant_data) in enumerate(results_dict.items()):
        # Calculate the 0-90 rise time for exactly the first 10 trials for each load condition
        light_load_0_90_rise_times = []
        heavy_load_0_90_rise_times = []
        light_trial_count = 0
        heavy_trial_count = 0
        
        for trial in participant_data['main']['runs']:
            if light_trial_count >= 10 and heavy_trial_count >= 10:
                break
            
            if trial.is_load_visible():
                position_over_threshold = trial[trial['EncoderRadians_smooth'] > 0.9]
                if not position_over_threshold.empty:
                    rise_time = position_over_threshold['elapsed_time'].iloc[0]
                else:
                    rise_time = np.nan
                
                if trial.get_load() == 0.1 and light_trial_count < 10:
                    light_load_0_90_rise_times.append(rise_time)
                    if not np.isnan(rise_time):
                        all_light_trials.append(light_trial_count)
                        all_light_times.append(rise_time)
                    light_trial_count += 1
                elif trial.get_load() == 0.5 and heavy_trial_count < 10:
                    heavy_load_0_90_rise_times.append(rise_time)
                    if not np.isnan(rise_time):
                        all_heavy_trials.append(heavy_trial_count)
                        all_heavy_times.append(rise_time)
                    heavy_trial_count += 1
        
        # Plot the data for this participant with a unique color - scatter points only
        ax1.scatter(range(len(light_load_0_90_rise_times)), light_load_0_90_rise_times, 
                   color=colors[i], alpha=0.8, s=100)
        
        ax2.scatter(range(len(heavy_load_0_90_rise_times)), heavy_load_0_90_rise_times, 
                   color=colors[i], alpha=0.8, s=100)
    
    # Fit exponential decay to the data: y = a * exp(-b * x) + c
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    from scipy.optimize import curve_fit
    
    # Fit light load data
    if all_light_trials and all_light_times:
        try:
            light_params, _ = curve_fit(exp_decay, all_light_trials, all_light_times, 
                                       p0=[1.0, 0.5, 0.5], maxfev=10000)
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(0, 9, 100)
            y_smooth = exp_decay(x_smooth, *light_params)
            
            # Plot the fitted curve with black dotted line
            ax1.plot(x_smooth, y_smooth, 'k--', linewidth=3, 
                    label=f"Fit: {light_params[0]:.2f} * exp(-{light_params[1]:.2f} * x) + {light_params[2]:.2f}")
            
            # Add legend
            ax1.legend(fontsize=12, loc='upper right')
        except:
            pass
    
    # Fit heavy load data
    if all_heavy_trials and all_heavy_times:
        try:
            heavy_params, _ = curve_fit(exp_decay, all_heavy_trials, all_heavy_times, 
                                       p0=[1.0, 0.5, 0.5], maxfev=10000)
            
            # Generate smooth curve for plotting
            x_smooth = np.linspace(0, 9, 100)
            y_smooth = exp_decay(x_smooth, *heavy_params)
            
            # Plot the fitted curve with black dotted line
            ax2.plot(x_smooth, y_smooth, 'k--', linewidth=3,
                    label=f"Fit: {heavy_params[0]:.2f} * exp(-{heavy_params[1]:.2f} * x) + {heavy_params[2]:.2f}")
            
            # Add legend
            ax2.legend(fontsize=12, loc='upper right')
        except:
            pass
    
    # Add labels with larger, bolder text
    ax1.set_xlabel('Trial Number', fontsize=16, fontweight='bold')
    ax1.set_ylabel('0-90 Rise Time (s)', fontsize=16, fontweight='bold')
    ax1.set_title('Light Load (0.1N-m) Rise Time Adaptation', fontsize=18, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True)
    
    ax2.set_xlabel('Trial Number', fontsize=16, fontweight='bold')
    ax2.set_ylabel('0-90 Rise Time (s)', fontsize=16, fontweight='bold')
    ax2.set_title('Heavy Load (0.5N-m) Rise Time Adaptation', fontsize=18, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.grid(True)
    
    if title: 
        fig.suptitle(title, fontsize=20, fontweight='bold')
    
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    
    if save_fig and save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

    
def plot_medium_adaptation(results_dict, title=None, save_fig=False, save_path=None): 
    """
    Plot the medium adaptation for a single participant. We are going to create a (3, 2) grid of subplots. 
    The left column shows position, grip force, and velocity timeseries for the last few trials 
    before the first visible medium is encountered and the first visible medium trial.
    The right column shows the corresponding virtual trajectories and velocities.
    """
    
    # Increase font size for all text elements
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    
    # Create figure with custom gridspec to handle different number of plots in each column
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.3, 0.3])
    
    # Create axes for left column (3 plots)
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_grip = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[2, 0])
    
    # Create axes for right column (2 plots that span the same height as the left column)
    ax_vpos = fig.add_subplot(gs[0:2, 1])  # Virtual position spans top 2/3
    ax_vvel = fig.add_subplot(gs[2, 1])    # Virtual velocity takes bottom 1/3
    
    # First, we need to find the last few trials before the first visible medium is encountered
    # and the first medium trial
    first_medium_trial = None
    last_trials_before_medium = []
    num_trials_to_average = 3  # Number of trials to average before medium
    
    for i, trial in enumerate(results_dict['main']['runs']):
        if trial.is_load_visible() and trial.get_load() == 0.3: 
            first_medium_trial = results_dict['main']['runs'][i]
            # Get the last few trials before medium
            for j in range(1, num_trials_to_average + 1):
                if i - j >= 0:  # Make sure we don't go out of bounds
                    last_trials_before_medium.append(results_dict['main']['runs'][i-j])
            break
    
    # Plot position data (left column)
    for i, trial in enumerate(last_trials_before_medium):
        # Convert numerical load to text
        load_text = "Light" if trial.get_load() == 0.1 else "Heavy" if trial.get_load() == 0.5 else f"{trial.get_load()}"
        load_label = f"Before Medium (Load: {load_text})" if i == 0 else None
        ax_pos.plot(trial['elapsed_time'], trial['EncoderRadians_smooth'], 
                 alpha=0.5, color='grey', label=load_label, linewidth=3)
    
    ax_pos.plot(first_medium_trial['elapsed_time'], first_medium_trial['EncoderRadians_smooth'], 
             color='green', linewidth=4, label='First Medium Trial')
    ax_pos.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax_pos.set_ylabel('Position (rad)', fontweight='bold', fontsize=14)
    ax_pos.legend(fontsize=12, prop={'weight': 'bold'})
    ax_pos.grid(True)
    ax_pos.tick_params(labelsize=12)
    
    # Plot grip force timeseries (left column)
    for i, trial in enumerate(last_trials_before_medium):
        ax_grip.plot(trial['elapsed_time'], trial['GripForce'], 
                 alpha=0.5, color='grey', label='Before Medium' if i == 0 else None, linewidth=3)
    
    ax_grip.plot(first_medium_trial['elapsed_time'], first_medium_trial['GripForce'], 
             color='green', linewidth=4, label='First Medium Trial')
    ax_grip.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax_grip.set_ylabel('Grip Force (N)', fontweight='bold', fontsize=14)
    ax_grip.legend(fontsize=12, prop={'weight': 'bold'})
    ax_grip.grid(True)
    ax_grip.tick_params(labelsize=12)
    
    # Plot velocity timeseries (left column)
    for i, trial in enumerate(last_trials_before_medium):
        ax_vel.plot(trial['elapsed_time'], trial['EncoderRadians_dot_smooth'], 
                 alpha=0.5, color='grey', label='Before Medium' if i == 0 else None, linewidth=3)
    
    ax_vel.plot(first_medium_trial['elapsed_time'], first_medium_trial['EncoderRadians_dot_smooth'], 
             color='green', linewidth=4, label='First Medium Trial')
    ax_vel.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
    ax_vel.set_ylabel('Velocity (rad/s)', fontweight='bold', fontsize=14)
    ax_vel.legend(fontsize=12, prop={'weight': 'bold'})
    ax_vel.grid(True)
    ax_vel.tick_params(labelsize=12)
    
    # Plot virtual trajectories (right column)
    # Check if virtual trajectories have been calculated
    has_virtual_trajectories = all('VirtualTrajectory' in trial.columns for trial in last_trials_before_medium + [first_medium_trial])
    
    if has_virtual_trajectories:
        # Plot virtual position (right column, top)
        for i, trial in enumerate(last_trials_before_medium):
            load_text = "Light" if trial.get_load() == 0.1 else "Heavy" if trial.get_load() == 0.5 else f"{trial.get_load()}"
            load_label = f"Before Medium (Load: {load_text})" if i == 0 else None
            ax_vpos.plot(trial['elapsed_time'], trial['VirtualTrajectory'], 
                     alpha=0.5, color='grey', label=load_label, linewidth=3)
        
        ax_vpos.plot(first_medium_trial['elapsed_time'], first_medium_trial['VirtualTrajectory'], 
                 color='green', linewidth=4, label='First Medium Trial')
        ax_vpos.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
        ax_vpos.set_ylabel('Virtual Position (rad)', fontweight='bold', fontsize=14)
        ax_vpos.legend(fontsize=12, prop={'weight': 'bold'})
        ax_vpos.grid(True)
        ax_vpos.tick_params(labelsize=12)
        
        # Plot virtual velocity (right column, bottom)
        for i, trial in enumerate(last_trials_before_medium):
            ax_vvel.plot(trial['elapsed_time'], trial['VirtualTrajectory_Dot'], 
                     alpha=0.5, color='grey', label='Before Medium' if i == 0 else None, linewidth=3)
        
        ax_vvel.plot(first_medium_trial['elapsed_time'], first_medium_trial['VirtualTrajectory_Dot'], 
                 color='green', linewidth=4, label='First Medium Trial')
        ax_vvel.set_xlabel('Time (s)', fontweight='bold', fontsize=14)
        ax_vvel.set_ylabel('Virtual Velocity (rad/s)', fontweight='bold', fontsize=14)
        ax_vvel.legend(fontsize=12, prop={'weight': 'bold'})
        ax_vvel.grid(True)
        ax_vvel.tick_params(labelsize=12)
    else:
        # If virtual trajectories haven't been calculated, display a message
        ax_vpos.text(0.5, 0.5, 'Virtual trajectories not calculated', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax_vpos.transAxes, fontsize=14, fontweight='bold')
        ax_vpos.set_xticks([])
        ax_vpos.set_yticks([])
        
        ax_vvel.text(0.5, 0.5, 'Virtual velocities not calculated', 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax_vvel.transAxes, fontsize=14, fontweight='bold')
        ax_vvel.set_xticks([])
        ax_vvel.set_yticks([])
    
    # Add a title indicating the load types of the trials before medium
    load_types = []
    for trial in last_trials_before_medium:
        load_value = trial.get_load()
        if load_value == 0.1:
            load_types.append("Light")
        elif load_value == 0.5:
            load_types.append("Heavy")
        else:
            load_types.append(f"{load_value}")
    
    load_info = f"Loads before medium: {', '.join(load_types)}"
    if title:
        plt.suptitle(f"{title}\n{load_info}", fontsize=16, fontweight='bold')
    else:
        plt.suptitle(load_info, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig and save_path: 
        plt.savefig(save_path, dpi=300)
    
    # Reset font parameters to default after creating this plot
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size'], 
                         'font.weight': plt.rcParamsDefault['font.weight']})
    
    return fig

def plot_medium_model_trajectories(results_dict, title=None, save_fig=False, save_path=None): 
    """
        We are going to create a 2x3 grid of subplots. First row is position, the second row is velocity. 
        We are going to plot the average trajectories for position and grip for heavy and light loads in columns 1 and 2. 
        We are going to do the same for column 3, though only for the first visible medium trial. 
        This is for a single participant.
    """
    
    # Create a 2x3 grid of subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    
    # Unpack axes for easier access
    (ax_heavy_pos, ax_light_pos, ax_medium_pos), (ax_heavy_grip, ax_light_grip, ax_medium_grip) = axes
    
    # Get the average trajectories for different load conditions
    average_heavy_trajectory = results_dict['main']['average_trajectories']['0.5_normal_visible']
    average_light_trajectory = results_dict['main']['average_trajectories']['0.1_normal_visible']
    medium_trajectory = results_dict['main']['average_trajectories']['first_medium_catch_visible']
    
    # Plot position data (top row)
    ax_heavy_pos.plot(average_heavy_trajectory['elapsed_time'], average_heavy_trajectory['EncoderRadians_smooth'], 
             label='Position', color='blue')
    ax_heavy_pos.set_title('Heavy Load (0.5)', fontsize=14)
    ax_heavy_pos.set_ylabel('Position (rad)', fontsize=12)
    ax_heavy_pos.grid(True)
    
    ax_light_pos.plot(average_light_trajectory['elapsed_time'], average_light_trajectory['EncoderRadians_smooth'], 
             label='Position', color='blue')
    ax_light_pos.set_title('Light Load (0.1)', fontsize=14)
    ax_light_pos.grid(True)
    
    ax_medium_pos.plot(medium_trajectory['elapsed_time'], medium_trajectory['EncoderRadians_smooth'], 
             label='Position', color='blue')
    ax_medium_pos.set_title('Medium Load (0.3)', fontsize=14)
    ax_medium_pos.grid(True)
    
    # Plot grip force data (bottom row)
    ax_heavy_grip.plot(average_heavy_trajectory['elapsed_time'], average_heavy_trajectory['GripForce'], 
             label='Grip Force', color='red')
    ax_heavy_grip.set_xlabel('Time (s)', fontsize=12)
    ax_heavy_grip.set_ylabel('Grip Force (N)', fontsize=12)
    ax_heavy_grip.grid(True)
    
    ax_light_grip.plot(average_light_trajectory['elapsed_time'], average_light_trajectory['GripForce'], 
             label='Grip Force', color='red')
    ax_light_grip.set_xlabel('Time (s)', fontsize=12)
    ax_light_grip.grid(True)
    
    ax_medium_grip.plot(medium_trajectory['elapsed_time'], medium_trajectory['GripForce'], 
             label='Grip Force', color='red')
    ax_medium_grip.set_xlabel('Time (s)', fontsize=12)
    ax_medium_grip.grid(True)
    
    # Add a title to the plot
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    if save_fig and save_path: 
        plt.savefig(save_path, dpi=300)
    
    return fig
    
def plot_combined_trials(data: pd.DataFrame, live_indices: list[tuple[int, int]], output_dir: str):
    """
    Plot all trials from live_indices on the same set of axes, overlaying the data.
    
    Args:
        data (pd.DataFrame): DataFrame containing all the trial data
        live_indices (list[tuple[int, int]]): List of (start, end) indices for each trial
        output_dir (str): Directory to save the output plot
    """
    # Create figure with subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12))
    
    # Plot each trial's data
    for i, (start, end) in enumerate(live_indices):
        # Calculate elapsed time for this trial
        elapsed_time = data['time_s'][start:end] - data['time_s'][start]
        
        # Plot position data
        ax1.plot(elapsed_time, data['EncoderRadians_smooth'][start:end], alpha=0.5, label=f'Trial {i+1}')
        
        # Plot velocity data
        ax2.plot(elapsed_time, data['EncoderRadians_dot_smooth'][start:end], alpha=0.5, label=f'Trial {i+1}')
        
        # Plot acceleration data
        ax3.plot(elapsed_time, data['EncoderRadians_ddot_smooth'][start:end], alpha=0.5, label=f'Trial {i+1}')
        
        # Plot torque data
        if 'AppliedTorque' in data: 
            ax4.plot(elapsed_time, data['AppliedTorque'][start:end], alpha=0.5, label=f'Trial {i+1}')
        else: 
            ax4.plot(elapsed_time, data['WallTorque'][start:end], alpha=0.5, label=f'Trial {i+1}')
        
        # Plot grip force data
        ax5.plot(elapsed_time, data['GripForce'][start:end], alpha=0.5, label=f'Trial {i+1}')
    
    # Set labels and titles
    ax1.set_ylabel('Position (rad)')
    ax1.set_title('Encoder Position - All Trials')
    ax1.grid(True)
    
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.set_title('Encoder Velocity - All Trials')
    ax2.grid(True)
    
    ax3.set_ylabel('Acceleration (rad/s²)')
    ax3.set_title('Encoder Acceleration - All Trials')
    ax3.grid(True)
    
    ax4.set_ylabel('Torque (Nm)')
    ax4.set_title('Applied Torque - All Trials')
    ax4.grid(True)
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Force (N)')
    ax5.set_title('Grip Force - All Trials')
    ax5.grid(True)
    ax5.legend()
    
    # Find the maximum duration among all trials to set consistent x-axis
    max_duration = max(data['time_s'][end] - data['time_s'][start] 
                      for start, end in live_indices)
    
    # Set x-axis ticks at 0.1 second intervals
    xticks = np.arange(0, max_duration + 0.1, 0.1)
    ax5.set_xticks(xticks)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    fig.savefig(os.path.join(output_dir, 'combined.png'))
    plt.close(fig)

    
def plot_group_impedance_components(results_dict: dict, save_fig: bool, save_path: str): 
    """
    Plot the impedance components for the group. We generate a 2x3 subplot. The top row is inertia, damping, and stiffness components 
    v.s. grip force threshold for each participant, with thier own colored line. The bottom row is the average of the components across participants. 
    With standard deivations shown. 
    """
    
    # Create figure with subplots - use nrows, ncols syntax to ensure proper unpacking
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    # Unpack axes properly for a 2x3 grid
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
    
    # Set font size and weight for better visibility in PDF
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    
    # Initialize arrays to store data for all participants
    thresholds = []
    inertias = np.empty((0, 0))
    dampings = np.empty((0, 0))
    stiffnesses = np.empty((0, 0))
    
    # Plot inertia components
    for participant_idx, participant in enumerate(results_dict):
        # Get the inertia, damping, and stiffness values for this participant
        participant_results = results_dict[participant]['calibration']['results']['by_grip_force_threshold']
        
        # Extract grip force thresholds for this participant
        participant_thresholds = np.array(list(participant_results.keys()))
        
        # If this is the first participant, initialize the arrays
        if participant_idx == 0:
            thresholds = participant_thresholds
            inertias = np.zeros((len(results_dict), len(thresholds)))
            dampings = np.zeros((len(results_dict), len(thresholds)))
            stiffnesses = np.zeros((len(results_dict), len(thresholds)))
        
        # Extract values for this participant
        for i, grip_force_threshold in enumerate(thresholds):
            if grip_force_threshold in participant_results:
                inertias[participant_idx, i] = participant_results[grip_force_threshold]['inertia']
                dampings[participant_idx, i] = participant_results[grip_force_threshold]['damping']
                stiffnesses[participant_idx, i] = participant_results[grip_force_threshold]['stiffness']
        
        # Use thicker lines and larger markers for better visibility
        ax1.scatter(thresholds, inertias[participant_idx], s=80)
        ax1.plot(thresholds, inertias[participant_idx], '-', linewidth=2.5)
        ax2.scatter(thresholds, dampings[participant_idx], s=80)
        ax2.plot(thresholds, dampings[participant_idx], '-', linewidth=2.5)
        ax3.scatter(thresholds, stiffnesses[participant_idx], s=80)
        ax3.plot(thresholds, stiffnesses[participant_idx], '-', linewidth=2.5)

    # Calculate averages for bottom row plots
    inertia_means = np.mean(inertias, axis=0)
    inertia_stds = np.std(inertias, axis=0)
    damping_means = np.mean(dampings, axis=0)
    damping_stds = np.std(dampings, axis=0)
    stiffness_means = np.mean(stiffnesses, axis=0)
    stiffness_stds = np.std(stiffnesses, axis=0)
    
    # Plot average inertia with error bars in bottom row - thicker lines and larger markers
    ax4.errorbar(thresholds, inertia_means, yerr=inertia_stds, fmt='-o', 
                 capsize=8, label='Average Inertia', ecolor='gray', 
                 linewidth=3, markersize=10)
    
    # Plot average damping with error bars in bottom row
    ax5.errorbar(thresholds, damping_means, yerr=damping_stds, fmt='-o', 
                 capsize=8, label='Average Damping', ecolor='gray',
                 linewidth=3, markersize=10)
    
    # Plot average stiffness with error bars in bottom row
    ax6.errorbar(thresholds, stiffness_means, yerr=stiffness_stds, fmt='-o', 
                 capsize=8, label='Average Stiffness', ecolor='gray',
                 linewidth=3, markersize=10)
            
    # Set labels and titles with larger, bolder font
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, linewidth=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
    
    ax1.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Inertia (kg*m²)', fontsize=14, fontweight='bold')
    ax1.set_title('Individual Inertia', fontsize=16, fontweight='bold')
    
    ax2.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Damping (Nm*s/rad)', fontsize=14, fontweight='bold')
    ax2.set_title('Individual Damping', fontsize=16, fontweight='bold')
    
    ax3.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Stiffness (Nm/rad)', fontsize=14, fontweight='bold')
    ax3.set_title('Individual Stiffness', fontsize=16, fontweight='bold')
    
    ax4.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Inertia (kg*m²)', fontsize=14, fontweight='bold')
    ax4.set_title('Average Inertia', fontsize=16, fontweight='bold')
    
    ax5.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Damping (Nm*s/rad)', fontsize=14, fontweight='bold')
    ax5.set_title('Average Damping', fontsize=16, fontweight='bold')
    
    ax6.set_xlabel('Grip Force Threshold (N)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Stiffness (Nm/rad)', fontsize=14, fontweight='bold')
    ax6.set_title('Average Stiffness', fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    if save_fig: 
        fig.savefig(save_path, dpi=300)
        
    # Reset font parameters to default after creating this plot
    plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size'], 
                         'font.weight': plt.rcParamsDefault['font.weight']})
    
    plt.close(fig)
    
    return fig



def plot_calibration_metrics(results: pd.DataFrame, output_dir: str): 
    # Plot the inertia, damping coefficient, and stiffness v.s. grip force 
    # across all trials in results. Trials is a column in results. GripThreshold is a column in results. 
    # Do a scatter plot, and the connect the data. Make a 1x3 subplot.
    
    # Sort results by grip force
    results = results.sort_values('Grip Force Threshold (N)')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot inertia 
    ax1.scatter(results['Grip Force Threshold (N)'], results['Estimated Inertia (kg*m^2)'])
    ax1.plot(results['Grip Force Threshold (N)'], results['Estimated Inertia (kg*m^2)'], '-')
    ax1.set_xlabel('Grip Force (N)')
    ax1.set_ylabel('Estimated Inertia (kg*m^2)')
    ax1.set_title('Inertia')
    ax1.grid(True)
    
    # Plot damping coefficient 
    ax2.scatter(results['Grip Force Threshold (N)'], results['Estimated Damping Constant (Nm*s/rad)'])
    ax2.plot(results['Grip Force Threshold (N)'], results['Estimated Damping Constant (Nm*s/rad)'], '-')
    ax2.set_xlabel('Grip Force (N)')
    ax2.set_ylabel('Estimated Damping Constant (Nm*s/rad)')
    ax2.set_title('Damping Coefficient')
    ax2.grid(True)
    
    # Plot stiffness 
    ax3.scatter(results['Grip Force Threshold (N)'], results['Estimated Stiffness (Nm/rad)'])
    ax3.plot(results['Grip Force Threshold (N)'], results['Estimated Stiffness (Nm/rad)'], '-')
    ax3.set_xlabel('Grip Force (N)')
    ax3.set_ylabel('Estimated Stiffness (Nm/rad)')
    ax3.set_title('Stiffness')
    ax3.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    fig.savefig(os.path.join(output_dir, 'calibration_metrics.png'))
    plt.close(fig)
    
def plot_aggregate_calibration_metrics(results_list: list[pd.DataFrame], output_dir: str):
    # Create figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a color for each dataset
    colors = plt.cm.rainbow(np.linspace(0, 1, len(results_list)))
    
    # Plot each set of results with different colors
    for i, (results, color) in enumerate(zip(results_list, colors)):
        # Sort results by grip force
        results = results.sort_values('Grip Force Threshold (N)')
        
        # Plot inertia 
        ax1.scatter(results['Grip Force Threshold (N)'], results['Estimated Inertia (kg*m^2)'], 
                   color=color, label=f'Trial {i}')
        ax1.plot(results['Grip Force Threshold (N)'], results['Estimated Inertia (kg*m^2)'], 
                '-', color=color)
        
        # Plot damping coefficient 
        ax2.scatter(results['Grip Force Threshold (N)'], results['Estimated Damping Constant (Nm*s/rad)'], 
                   color=color, label=f'Trial {i}')
        ax2.plot(results['Grip Force Threshold (N)'], results['Estimated Damping Constant (Nm*s/rad)'], 
                '-', color=color)
        
        # Plot stiffness 
        ax3.scatter(results['Grip Force Threshold (N)'], results['Estimated Stiffness (Nm/rad)'], 
                   color=color, label=f'Trial {i}')
        ax3.plot(results['Grip Force Threshold (N)'], results['Estimated Stiffness (Nm/rad)'], 
                '-', color=color)
    
    # Set labels and titles
    ax1.set_xlabel('Grip Force (N)')
    ax1.set_ylabel('Estimated Inertia (kg*m^2)')
    ax1.set_title('Inertia')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Grip Force (N)')
    ax2.set_ylabel('Estimated Damping Constant (Nm*s/rad)')
    ax2.set_title('Damping Coefficient')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_xlabel('Grip Force (N)')
    ax3.set_ylabel('Estimated Stiffness (Nm/rad)')
    ax3.set_title('Stiffness')
    ax3.grid(True)
    ax3.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save and close
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, 'aggregate_calibration_metrics.png'))
    plt.close(fig)


def plot_calibration_results_and_trials(results, participant, output_dir=None): 
    """
    Plot calibration results and trials for a given participant.
    Creates separate figures showing the calibration metrics over trials
    and the time series data from each trial.
    """
    print(f"Plotting calibration results for participant: {participant}")
    calibration_runs = results[participant]['calibration']
    
    plt.rcParams.update({'font.size': 16})
    
    # Create figure for calibration metrics
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(12, 10)) 
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(figsize=(12, 10))
    ax4 = fig4.add_subplot(111)
    
    # Plot metrics for each calibration run
    for run_idx, run in calibration_runs.items():
        metrics_df = run['metrics']
        
        # Plot with both scatter and lines
        ax1.scatter(metrics_df['Grip Force Threshold'], metrics_df['Inertia'], 
                   label=f'Run {run_idx}', s=100)
        ax1.plot(metrics_df['Grip Force Threshold'], metrics_df['Inertia'], 
                '-', alpha=0.8, linewidth=3)
        
        ax2.scatter(metrics_df['Grip Force Threshold'], metrics_df['Damping'], 
                   label=f'Run {run_idx}', s=100)
        ax2.plot(metrics_df['Grip Force Threshold'], metrics_df['Damping'], 
                '-', alpha=0.8, linewidth=3)
        
        ax3.scatter(metrics_df['Grip Force Threshold'], metrics_df['Stiffness'], 
                   label=f'Run {run_idx}', s=100)
        ax3.plot(metrics_df['Grip Force Threshold'], metrics_df['Stiffness'], 
                '-', alpha=0.8, linewidth=3)
        
        ax4.scatter(metrics_df['Grip Force Threshold'], metrics_df['Residual Error'], 
                   label=f'Run {run_idx}', s=100)
        ax4.plot(metrics_df['Grip Force Threshold'], metrics_df['Residual Error'], 
                '-', alpha=0.8, linewidth=3)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, linewidth=1.5)
        ax.tick_params(width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.legend(fontsize=14)
    
    ax1.set_xlabel('Grip Force (N)', fontsize=18)
    ax1.set_ylabel('Inertia (kg*m²)', fontsize=18)
    ax1.set_title('Inertia vs Grip Force', fontsize=20)
    
    ax2.set_xlabel('Grip Force (N)', fontsize=18)
    ax2.set_ylabel('Damping (Nm*s/rad)', fontsize=18)
    ax2.set_title('Damping vs Grip Force', fontsize=20)
    
    ax3.set_xlabel('Grip Force (N)', fontsize=18)
    ax3.set_ylabel('Stiffness (Nm/rad)', fontsize=18)
    ax3.set_title('Stiffness vs Grip Force', fontsize=20)
    
    ax4.set_xlabel('Grip Force (N)', fontsize=18)
    ax4.set_ylabel('Residual Error', fontsize=18)
    ax4.set_title('Residual Error vs Grip Force', fontsize=20)
    
    # Create separate figures for time series data
    fig5 = plt.figure(figsize=(12, 8))
    ax5 = fig5.add_subplot(111)
    fig6 = plt.figure(figsize=(12, 8))
    ax6 = fig6.add_subplot(111)
    fig7 = plt.figure(figsize=(12, 8))
    ax7 = fig7.add_subplot(111)
    
    # Plot time series for each valid trial
    for run_idx, run in calibration_runs.items():
        for trial_idx, trial in run['trials'].items():
            if trial['is_valid']:
                trial_data = trial['data']
                time = trial_data['elapsed_time']
                
                ax5.plot(time, trial_data['EncoderRadians_smooth'], 
                        alpha=0.8, linewidth=2, label=f'Run {run_idx} Trial {trial_idx}')
                ax6.plot(time, trial_data['EncoderRadians_dot_smooth'], 
                        alpha=0.8, linewidth=2, label=f'Run {run_idx} Trial {trial_idx}')
                ax7.plot(time, trial_data['GripForce'], 
                        alpha=0.8, linewidth=2, label=f'Run {run_idx} Trial {trial_idx}')
    
    for ax in [ax5, ax6, ax7]:
        ax.grid(True, linewidth=1.5)
        ax.tick_params(width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.legend(fontsize=14)
    
    ax5.set_ylabel('Position (rad)', fontsize=18)
    ax5.set_title('Encoder Position', fontsize=20)
    
    ax6.set_ylabel('Velocity (rad/s)', fontsize=18)
    ax6.set_title('Encoder Velocity', fontsize=20)
    
    ax7.set_ylabel('Force (N)', fontsize=18)
    ax7.set_xlabel('Time (s)', fontsize=18)
    ax7.set_title('Grip Force', fontsize=20)
    
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        fig.tight_layout()
    
    plt.show()

    
def plot_calibration_results_for_paper(results, output_dir=None): 
    """
    Plot calibration results for the paper. This function takes a results 
    dictionary, and plots the following figures:

    - Individual participant plots for Inertia, Damping, and Stiffness vs grip force threshold
    - Group mean plots for Inertia, Damping, and Stiffness vs grip force threshold with standard deviations
    """
    
    plt.rcParams.update({'font.size': 18})
    
    # Create 6 separate figures
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(12, 8)) 
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure(figsize=(12, 8))
    ax3 = fig3.add_subplot(111)
    fig4 = plt.figure(figsize=(12, 8))
    ax4 = fig4.add_subplot(111)
    fig5 = plt.figure(figsize=(12, 8))
    ax5 = fig5.add_subplot(111)
    fig6 = plt.figure(figsize=(12, 8))
    ax6 = fig6.add_subplot(111)

    # With the same columns as the metrics dataframe.
    all_results_df = pd.DataFrame(columns=['participant', 'inertia', 'damping', 'stiffness', 'grip_force_threshold'])
    
    # Create dataframe for fit parameters
    fit_params_df = pd.DataFrame(columns=['participant', 'participant_idx', 'inertia_slope', 'inertia_intercept',
                                        'damping_slope', 'damping_intercept',
                                        'stiffness_slope', 'stiffness_intercept',
                                        'average_residual_error'])
    
    print("Processing data for each participant...")

    # First get the mean and std for each trial by participant. 
    for participant in results.keys(): 
        print(f"\nProcessing participant: {participant}")
        
        # Pick a color for this participant based on their index in the results dictionary
        participant_idx = list(results.keys()).index(participant)
        color = plt.cm.tab10(participant_idx % 10)
        
        # Get all calibration data for this participant
        single_participant_df = pd.DataFrame()
        for run_idx, run in results[participant]['calibration'].items():
            # Convert metrics dictionary to DataFrame
            metrics = []
            for trial in run['trials'].values():
                if trial['is_valid'] and trial['metrics']['grip_force_threshold'] < 30:
                    metrics.append({
                        'inertia': trial['metrics']['inertia'],
                        'damping': trial['metrics']['damping'],
                        'stiffness': trial['metrics']['stiffness'],
                        'grip_force_threshold': trial['metrics']['grip_force_threshold'],
                        'residual_error': trial['metrics']['residual_error']
                    })
            run_df = pd.DataFrame(metrics)
            single_participant_df = pd.concat([single_participant_df, run_df], ignore_index=True)
        
        print(f"Number of valid trials for {participant}: {len(single_participant_df)}")
        if len(single_participant_df) == 0:
            print(f"Warning: No valid trials found for {participant}")
            continue
            
        # Calculate means for each grip force threshold
        mean_results_df = single_participant_df.groupby('grip_force_threshold').mean().reset_index()
        
        # Calculate linear fits
        grip = mean_results_df['grip_force_threshold']
        j1, j0 = np.polyfit(grip, mean_results_df['inertia'], 1)
        b1, b0 = np.polyfit(grip, mean_results_df['damping'], 1)
        k1, k0 = np.polyfit(grip, mean_results_df['stiffness'], 1)

        # Calculate residuals manually
        j_residuals = np.sum((mean_results_df['inertia'] - (j1*grip + j0))**2)
        b_residuals = np.sum((mean_results_df['damping'] - (b1*grip + b0))**2) 
        k_residuals = np.sum((mean_results_df['stiffness'] - (k1*grip + k0))**2)

        print(f"For participant {participant}, the residual errors are:")
        print(f"Inertia: {j_residuals:.4f}") 
        print(f"Damping: {b_residuals:.4f}")
        print(f"Stiffness: {k_residuals:.4f}")
        
        
        # Add fit parameters to dataframe
        new_fit_row = pd.DataFrame({
            'participant': [participant],
            'participant_idx': [participant_idx],
            'inertia_slope': [j1],
            'inertia_intercept': [j0],
            'damping_slope': [b1],
            'damping_intercept': [b0],
            'stiffness_slope': [k1],
            'stiffness_intercept': [k0], 
            'average_residual_error': [np.mean([j_residuals, b_residuals, k_residuals])]
        })
        fit_params_df = pd.concat([fit_params_df, new_fit_row], ignore_index=True)
        
        # Add to all results DataFrame
        for _, row in mean_results_df.iterrows():
            new_row = pd.DataFrame({
                'participant': [participant],
                'inertia': [row['inertia']], 
                'damping': [row['damping']],
                'stiffness': [row['stiffness']],
                'grip_force_threshold': [row['grip_force_threshold']]
            })
            all_results_df = pd.concat([all_results_df, new_row], ignore_index=True)
        
        # Plot individual participant data with thicker lines
        ax1.scatter(mean_results_df['grip_force_threshold'], mean_results_df['inertia']*10**4, 
                    color=color, s=150, label=f'P{participant_idx+1}')
        ax1.plot(mean_results_df['grip_force_threshold'], mean_results_df['inertia']*10**4, 
                '-', color=color, linewidth=4)
        
        ax2.scatter(mean_results_df['grip_force_threshold'], mean_results_df['damping'], 
                    color=color, s=150, label=f'P{participant_idx+1}')
        ax2.plot(mean_results_df['grip_force_threshold'], mean_results_df['damping'], 
                '-', color=color, linewidth=4)
        
        ax3.scatter(mean_results_df['grip_force_threshold'], mean_results_df['stiffness'], 
                    color=color, s=150, label=f'P{participant_idx+1}')
        ax3.plot(mean_results_df['grip_force_threshold'], mean_results_df['stiffness'], 
                '-', color=color, linewidth=4)

    # Calculate group statistics
    group_stats = all_results_df.groupby('grip_force_threshold').agg({
        'inertia': ['mean', 'std'],
        'damping': ['mean', 'std'],
        'stiffness': ['mean', 'std']
    }).reset_index()

    # Calculate linear fits for j, b, and k from all_results_df, and get residuals
    grip = all_results_df['grip_force_threshold']
    j1, j0 = np.polyfit(grip, all_results_df['inertia'], 1)
    b1, b0 = np.polyfit(grip, all_results_df['damping'], 1)
    k1, k0 = np.polyfit(grip, all_results_df['stiffness'], 1)
    
    j_residuals = np.sum((all_results_df['inertia'] - (j1*grip + j0))**2)
    b_residuals = np.sum((all_results_df['damping'] - (b1*grip + b0))**2) 
    k_residuals = np.sum((all_results_df['stiffness'] - (k1*grip + k0))**2)

    print(f"For the group, the residual errors are:")
    print(f"Inertia: {j_residuals:.4f}") 
    print(f"Damping: {b_residuals:.4f}")
    print(f"Stiffness: {k_residuals:.4f}")

    # Add to fit_params_df
    new_fit_row = pd.DataFrame({
        'participant': ['Group'],
        'participant_idx': [-1],
        'inertia_slope': [j1],
        'inertia_intercept': [j0],
        'damping_slope': [b1],
        'damping_intercept': [b0],
        'stiffness_slope': [k1],
        'stiffness_intercept': [k0], 
        'average_residual_error': [np.mean([j_residuals, b_residuals, k_residuals])]
    })
    fit_params_df = pd.concat([fit_params_df, new_fit_row], ignore_index=True)

    
    # Plot group means and standard deviations with thicker lines
    ax4.errorbar(group_stats['grip_force_threshold'], 
                group_stats['inertia']['mean']*10**4,
                yerr=group_stats['inertia']['std']*10**4,
                color='k', capsize=8, capthick=3, linewidth=4, label='Group Mean ± SD')
    ax4.scatter(group_stats['grip_force_threshold'], 
                group_stats['inertia']['mean']*10**4,
                color='k', s=150)
    
    ax5.errorbar(group_stats['grip_force_threshold'], 
                group_stats['damping']['mean'],
                yerr=group_stats['damping']['std'],
                color='k', capsize=8, capthick=3, linewidth=4, label='Group Mean ± SD')
    ax5.scatter(group_stats['grip_force_threshold'], 
                group_stats['damping']['mean'],
                color='k', s=150)
    
    ax6.errorbar(group_stats['grip_force_threshold'], 
                group_stats['stiffness']['mean'],
                yerr=group_stats['stiffness']['std'],
                color='k', capsize=8, capthick=3, linewidth=4, label='Group Mean ± SD')
    ax6.scatter(group_stats['grip_force_threshold'], 
                group_stats['stiffness']['mean'],
                color='k', s=150)
    
    # Set labels and titles with larger fonts
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.grid(True, linewidth=2)
        ax.tick_params(labelsize=16, width=2, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_xticks(np.arange(0, 31, 5))
    
    # Add legends with fancy boxes for individual plots
    for ax in [ax1, ax2, ax3]:
        legend = ax.legend(bbox_to_anchor=(0.05, 0.93, 0.9, 0.5), loc="lower left",
                          mode="expand", borderaxespad=0, ncol=len(results),
                          fancybox=True, shadow=True, fontsize=18)
        legend.get_frame().set_linewidth(2)
    
    ax1.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax1.set_ylabel('Inertia (×10⁻⁴ kg⋅m²)', fontsize=20)
    ax1.set_title('Individual Inertia', fontsize=24)
    ax1.set_ylim(0, 12)

    ax2.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax2.set_ylabel('Damping (N⋅m⋅s/rad)', fontsize=20)
    ax2.set_title('Individual Damping', fontsize=24)
    ax2.set_ylim(0, 0.1)

    ax3.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax3.set_ylabel('Stiffness (N⋅m/rad)', fontsize=20)
    ax3.set_title('Individual Stiffness', fontsize=24)
    ax3.set_ylim(0, 4)
    
    ax4.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax4.set_ylabel('Inertia (×10⁻⁴ kg⋅m²)', fontsize=20)
    ax4.set_title('Group Mean Inertia', fontsize=24)
    ax4.set_ylim(0, 12)
    ax4.legend(fontsize=16)
    
    ax5.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax5.set_ylabel('Damping (N⋅m⋅s/rad)', fontsize=20)
    ax5.set_title('Group Mean Damping', fontsize=24)
    ax5.set_ylim(0, 0.1)
    ax5.legend(fontsize=16)
    
    ax6.set_xlabel('Grip Force Threshold (N)', fontsize=20)
    ax6.set_ylabel('Stiffness (N⋅m/rad)', fontsize=20)
    ax6.set_title('Group Mean Stiffness', fontsize=24)
    ax6.set_ylim(0, 4)
    ax6.legend(fontsize=16)
    
    # Adjust layout for each figure
    for fig in [fig1, fig2, fig3, fig4, fig5, fig6]:
        fig.tight_layout()
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig1.savefig(os.path.join(output_dir, 'individual_inertia.png'))
        fig2.savefig(os.path.join(output_dir, 'individual_damping.png'))
        fig3.savefig(os.path.join(output_dir, 'individual_stiffness.png'))
        fig4.savefig(os.path.join(output_dir, 'group_inertia.png'))
        fig5.savefig(os.path.join(output_dir, 'group_damping.png'))
        fig6.savefig(os.path.join(output_dir, 'group_stiffness.png'))

        
    print(f"Showing results of linear fits: ")
    print(fit_params_df)
    
    plt.show()
    

    


    
def plot_main_trial_position_and_grip(results, participant, output_dir=None): 
    """
    Plot the position and grip force for all main trials for a given participant. 
    This function produces several figures:
    1. Individual grip force traces
    2. Individual position traces
    3. Velocity traces
    4. Average grip force traces with standard deviation bands
    5. Average position traces with standard deviation bands
    """
    print(f"Plotting main trials for participant: {participant}")
    main_runs = results[participant]['main']
    num_main_runs = results[participant]['num_main_runs']
    print(f"Number of main runs: {num_main_runs}")
    
    # Get number of each load case across all runs 
    load_counts = {}
    for run in main_runs.values(): 
        for trial in run['trials'].values():
            if not trial.get('is_valid', True):  # Skip invalid trials
                continue
            load = trial['load']
            if load in load_counts: 
                load_counts[load] += 1
            else: 
                load_counts[load] = 1
                
    print(f"Load counts: {load_counts}")
    
    # Split trials into different types using the processing function
    trial_types = split_runs_into_trial_types(main_runs)
    light_trials = trial_types['light']
    heavy_trials = trial_types['heavy']
    light_heavy_trials = trial_types['light_heavy']
    heavy_light_trials = trial_types['heavy_light']
            
    # Create separate figures
    fig1, ax1 = plt.subplots(figsize=(12, 6))  # Individual grip force
    fig2, ax2 = plt.subplots(figsize=(12, 6))  # Individual position
    fig3, ax3 = plt.subplots(figsize=(12, 6))  # Velocity
    fig4, ax4 = plt.subplots(figsize=(12, 6))  # Average grip force
    fig5, ax5 = plt.subplots(figsize=(12, 6))  # Average position

    # Only proceed if we have at least one trial
    if not light_trials and not heavy_trials:
        print("No valid trials found")
        return
        
    # Find first non-empty trial list to get time array
    first_trial = None
    for trial_list in [light_trials, heavy_trials, light_heavy_trials, heavy_light_trials]:
        if trial_list:
            first_trial = trial_list[0]
            break
            
    if first_trial is None:
        print("No valid trials found")
        return
        
    # Find index nearest to 0.3s for the first trial
    shared_final_index = np.argmin(np.abs(first_trial['elapsed_time'] - 0.3))
    time_array = first_trial['elapsed_time'].iloc[0:shared_final_index].values

    def plot_trials(trials, color, label):
        if not trials:
            return
            
        for i, trial in enumerate(trials):
            if i == 0:  # Only add label for first trial
                ax2.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['EncoderRadians_smooth'].iloc[:shared_final_index], f'{color}-', label=label, linewidth=2.5)
                ax3.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['EncoderRadians_dot_smooth'].iloc[:shared_final_index], f'{color}-', label=label, linewidth=2.5)
                ax1.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['GripForce'].iloc[:shared_final_index], f'{color}-', label=label, linewidth=2.5)
            else:
                ax2.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['EncoderRadians_smooth'].iloc[:shared_final_index], f'{color}-', linewidth=2.5)
                ax3.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['EncoderRadians_dot_smooth'].iloc[:shared_final_index], f'{color}-', linewidth=2.5)
                ax1.plot(trial['elapsed_time'].iloc[:shared_final_index], trial['GripForce'].iloc[:shared_final_index], f'{color}-', linewidth=2.5)

    def plot_average(trials, color):
        if not trials:
            return
            
        pos_array = np.array([trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values for trial in trials])
        grip_array = np.array([trial['GripForce'].iloc[0:shared_final_index].values for trial in trials])
        
        avg_pos = np.mean(pos_array, axis=0)
        std_pos = np.std(pos_array, axis=0)
        avg_grip = np.mean(grip_array, axis=0)
        std_grip = np.std(grip_array, axis=0)
        
        ax5.plot(time_array, avg_pos, f'{color}-', linewidth=3)
        ax5.fill_between(time_array, avg_pos - std_pos, avg_pos + std_pos, color=color, alpha=0.2)
        
        ax4.plot(time_array, avg_grip, f'{color}-', linewidth=3)
        ax4.fill_between(time_array, avg_grip - std_grip, avg_grip + std_grip, color=color, alpha=0.2)

    # Plot individual trials and averages
    plot_trials(light_trials, 'k', 'Light')
    plot_trials(heavy_trials, 'b', 'Heavy')  
    plot_trials(light_heavy_trials, 'g', 'Light->Heavy')
    plot_trials(heavy_light_trials, 'r', 'Heavy->Light')
    
    plot_average(light_trials, 'k')
    plot_average(heavy_trials, 'b')
    plot_average(light_heavy_trials, 'g')
    plot_average(heavy_light_trials, 'r')

    # Add axis labels and formatting
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.grid(True, linewidth=1.5)
        ax.tick_params(labelsize=14, width=2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.set_xlabel('Time (s)', labelpad=10, fontsize=18, fontweight='bold')
        
    ax1.set_ylabel('Grip Force (N)', labelpad=10, fontsize=18, fontweight='bold')
    ax1.set_title('Individual Grip Force Traces', fontsize=20, fontweight='bold', pad=15)
    
    ax2.set_ylabel('Position (rad)', labelpad=10, fontsize=18, fontweight='bold')
    ax2.set_title('Individual Position Traces', fontsize=20, fontweight='bold', pad=15)
    
    ax3.set_ylabel('Velocity (rad/s)', labelpad=10, fontsize=18, fontweight='bold')
    ax3.set_title('Individual Velocity Traces', fontsize=20, fontweight='bold', pad=15)
    
    ax4.set_ylabel('Grip Force (N)', labelpad=10, fontsize=18, fontweight='bold')
    ax4.set_title('Average Grip Force with Standard Deviation', fontsize=20, fontweight='bold', pad=15)
    
    ax5.set_ylabel('Position (rad)', labelpad=10, fontsize=18, fontweight='bold')
    ax5.set_title('Average Position with Standard Deviation', fontsize=20, fontweight='bold', pad=15)

    # # Add legends
    # for ax in [ax1, ax2, ax3, ax4, ax5]:
    #     handles, labels = ax.get_legend_handles_labels()
    #     if handles and labels:
    #         ax.legend(fontsize=16)

    # Adjust layouts
    for fig in [fig1, fig2, fig3, fig4, fig5]:
        fig.tight_layout(pad=2.0)
    
    plt.show()

def plot_virtual_trajectories_for_paper(results, participant, output_dir=None, title=None, save_fig=False, save_path=None): 
    """
    Plot the virtual trajectories for a given participant. 
    
    Args:
        results: Dictionary containing all participant results
        participant: Participant ID
        output_dir: Directory to save output files (optional)
        title: Optional title for the figure
        save_fig: Whether to save the figure
        save_path: Path to save the figure if save_fig is True
        
    Returns:
        None
    """
    
    # To do this, first we need to get the impedance v.s. grip force threshold data, as calculated in. 
    # plot_calibration_results_for_paper. 
    # Then, we need to get the main trials by load case as in plot_main_trial_position_and_grip. 
    # Then, we can plot the virtual trajectories for each load case.
    
    print(f"Plotting virtual trajectories for participant: {participant}")
    
    # Get calibration data and calculate impedance parameters vs grip force
    calibration_runs = results[participant]['calibration']
    single_participant_df = pd.DataFrame()
    
    # Collect valid calibration trials
    for run_idx, run in calibration_runs.items():
        metrics = []
        for trial in run['trials'].values():
            if trial['is_valid'] and trial['metrics']['grip_force_threshold'] < 30:
                metrics.append({
                    'inertia': trial['metrics']['inertia'],
                    'damping': trial['metrics']['damping'], 
                    'stiffness': trial['metrics']['stiffness'],
                    'grip_force_threshold': trial['metrics']['grip_force_threshold']
                })
        run_df = pd.DataFrame(metrics)
        single_participant_df = pd.concat([single_participant_df, run_df], ignore_index=True)

    # Calculate mean parameters for each grip force threshold
    mean_results_df = single_participant_df.groupby('grip_force_threshold').mean().reset_index()
    
    # Calculate linear fits for impedance parameters
    grip = mean_results_df['grip_force_threshold']
    j1, j0 = np.polyfit(grip, mean_results_df['inertia'], 1)
    b1, b0 = np.polyfit(grip, mean_results_df['damping'], 1)
    k1, k0 = np.polyfit(grip, mean_results_df['stiffness'], 1)
    
    # Get main trials split by load case
    main_runs = results[participant]['main']
            
    # Split into trial types
    trial_types = split_runs_into_trial_types(main_runs)
    light_trials = trial_types['light']
    heavy_trials = trial_types['heavy']
    light_heavy_trials = trial_types['light_heavy']
    heavy_light_trials = trial_types['heavy_light']
    
    print(f"Invalid trials: {trial_types['invalid_trials']}")
    
    print(f"Light trials: {len(light_trials)}, Heavy trials: {len(heavy_trials)}, Light->Heavy trials: {len(light_heavy_trials)}, Heavy->Light trials: {len(heavy_light_trials)}")
    
    # Now, we get the average trajectory for each load case
        # Find first non-empty trial list to get time array
    first_trial = None
    for trial_list in [light_trials, heavy_trials, light_heavy_trials, heavy_light_trials]:
        if trial_list:
            first_trial = trial_list[0]
            break
            
    if first_trial is None:
        print("No valid trials found")
        return
        
    # Find index nearest to 0.3s for the first trial
    shared_final_index = np.argmin(np.abs(first_trial['elapsed_time'] - 0.3))
    time_array = first_trial['elapsed_time'].iloc[0:shared_final_index].values
    
    # Initialize arrays to store average trajectories
    avg_pos_light = np.zeros_like(time_array)
    avg_vel_light = np.zeros_like(time_array)
    avg_accel_light = np.zeros_like(time_array)
    avg_pos_heavy = np.zeros_like(time_array)
    avg_vel_heavy = np.zeros_like(time_array)
    avg_accel_heavy = np.zeros_like(time_array)
    avg_grip_light = np.zeros_like(time_array)
    avg_grip_heavy = np.zeros_like(time_array)
    avg_torque_light = np.zeros_like(time_array)
    avg_torque_heavy = np.zeros_like(time_array)
    avg_pos_light_heavy = np.zeros_like(time_array)
    avg_pos_heavy_light = np.zeros_like(time_array)
    avg_grip_light_heavy = np.zeros_like(time_array)
    avg_grip_heavy_light = np.zeros_like(time_array)
    
    # Count number of valid trials for each case
    n_light = 0
    n_heavy = 0
    n_light_heavy = 0
    n_heavy_light = 0
    
    # Sum up trajectories for light trials
    for trial in light_trials:
        pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
        vel = trial['EncoderRadians_dot_smooth'].iloc[0:shared_final_index].values
        accel = trial['EncoderRadians_ddot_smooth'].iloc[0:shared_final_index].values
        grip = trial['GripForce'].iloc[0:shared_final_index].values
        torque = trial['CommandTorque'].iloc[0:shared_final_index].values
        avg_pos_light += pos
        avg_vel_light += vel
        avg_accel_light += accel
        avg_grip_light += grip
        avg_torque_light += torque
        n_light += 1
        
    # Sum up trajectories for heavy trials  
    for trial in heavy_trials:
        pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
        vel = trial['EncoderRadians_dot_smooth'].iloc[0:shared_final_index].values
        accel = trial['EncoderRadians_ddot_smooth'].iloc[0:shared_final_index].values
        grip = trial['GripForce'].iloc[0:shared_final_index].values
        torque = trial['CommandTorque'].iloc[0:shared_final_index].values
        avg_pos_heavy += pos
        avg_vel_heavy += vel
        avg_accel_heavy += accel
        avg_grip_heavy += grip
        avg_torque_heavy += torque
        n_heavy += 1
        
    for trial in light_heavy_trials:
        pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
        grip = trial['GripForce'].iloc[0:shared_final_index].values
        avg_pos_light_heavy += pos
        avg_grip_light_heavy += grip
        n_light_heavy += 1
        
    for trial in heavy_light_trials:
        pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
        grip = trial['GripForce'].iloc[0:shared_final_index].values
        avg_pos_heavy_light += pos
        avg_grip_heavy_light += grip
        n_heavy_light += 1
        
    # Calculate averages
    if n_light > 0:
        avg_pos_light /= n_light
        avg_vel_light /= n_light
        avg_accel_light /= n_light
        avg_grip_light /= n_light
        avg_torque_light /= n_light
    
    if n_heavy > 0:
        avg_pos_heavy /= n_heavy
        avg_vel_heavy /= n_heavy
        avg_accel_heavy /= n_heavy
        avg_grip_heavy /= n_heavy
        avg_torque_heavy /= n_heavy
    
    if n_light_heavy > 0:
        avg_pos_light_heavy /= n_light_heavy
        avg_grip_light_heavy /= n_light_heavy
        
    if n_heavy_light > 0:
        avg_pos_heavy_light /= n_heavy_light
        avg_grip_heavy_light /= n_heavy_light
    # To get the virtual trajectories, we need to use the impedance parameters based on the average
    # grip force in the first 70ms. 
    # avg_grip_force_light = np.mean(avg_grip_light[0:np.argmin(np.abs(time_array - 0.07))])
    # avg_grip_force_heavy = np.mean(avg_grip_heavy[0:np.argmin(np.abs(time_array - 0.07))])
    # avg_grip_force_light = np.mean(avg_grip_light)
    # avg_grip_force_heavy = np.mean(avg_grip_heavy)
    
    # Then we need to find the corresponding impedance values for each grip force threshold
    # using the linear fits we calculated earlier. 
    # j_light = j1 * avg_grip_force_light + j0
    # b_light = b1 * avg_grip_force_light + b0
    # k_light = k1 * avg_grip_force_light + k0
    
    # j_heavy = j1 * avg_grip_force_heavy + j0
    # b_heavy = b1 * avg_grip_force_heavy + b0
    # k_heavy = k1 * avg_grip_force_heavy + k0
    
    # Then to find the virtual trajectories, we need to solve a differntial equation: 
    # phi_dot = -(k/b)*phi + (1/b)*(k*theta + b*theta_dot + J*theta_ddot + CommandTorque)
    # We can do this using scipy's solve_ivp function.
    def phi_dot_light(t, y): 
        theta = avg_pos_light[np.argmin(np.abs(time_array - t))]
        theta_dot = avg_vel_light[np.argmin(np.abs(time_array - t))]
        theta_ddot = avg_accel_light[np.argmin(np.abs(time_array - t))]
        # command_torque = avg_torque_light[np.argmin(np.abs(time_array - t))]
        command_torque = np.clip(theta*(0.1/0.1), 0, 0.1)
        grip = avg_grip_light[np.argmin(np.abs(time_array - t))]
        j_light = j1 * grip + j0
        b_light = b1 * grip + b0
        k_light = k1 * grip + k0
        return -(k_light/b_light)*y[0] + (1/b_light)*(k_light*theta + b_light*theta_dot + j_light*theta_ddot + command_torque)
    
    def phi_dot_heavy(t, y): 
        theta = avg_pos_heavy[np.argmin(np.abs(time_array - t))]
        theta_dot = avg_vel_heavy[np.argmin(np.abs(time_array - t))]
        theta_ddot = avg_accel_heavy[np.argmin(np.abs(time_array - t))]
        grip = avg_grip_heavy[np.argmin(np.abs(time_array - t))]
        # command_torque = avg_torque_heavy[np.argmin(np.abs(time_array - t))]
        command_torque = np.clip(theta*(0.4/0.1), 0, 0.4)
        j_heavy = j1 * grip + j0
        b_heavy = b1 * grip + b0
        k_heavy = k1 * grip + k0
        return -(k_heavy/b_heavy)*y[0] + (1/b_heavy)*(k_heavy*theta + b_heavy*theta_dot + j_heavy*theta_ddot + command_torque)
    
    phi_light = solve_ivp(phi_dot_light, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
    phi_heavy = solve_ivp(phi_dot_heavy, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
    
    phi_light = np.array(phi_light.y[0])
    phi_heavy = np.array(phi_heavy.y[0])
    phi_light_dot = np.array([phi_dot_light(t, [phi_light[i]]) for i, t in enumerate(time_array)])
    phi_heavy_dot = np.array([phi_dot_heavy(t, [phi_heavy[i]]) for i, t in enumerate(time_array)])
    
    # Plot virtual trajectories
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(time_array, phi_light, 'k:', label='Light Virtual', linewidth=6)
    ax1.plot(time_array, phi_heavy, 'b:', label='Heavy Virtual', linewidth=6) 
    # ax1.legend(fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Control Input($\phi$) (rad)', fontsize=18, fontweight='bold')
    ax1.set_title('Control Input Estimate v.s. Time', fontsize=20, fontweight='bold')
    ax1.grid(True)
    ax1.set_xlim(-0.1, time_array[-1])
    ax1.tick_params(labelsize=16)
    
    # Plot actual trajectories
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(time_array, avg_pos_light, 'k-', label='Light', linewidth=4)
    ax2.plot(time_array, avg_pos_heavy, 'b-', label='Heavy', linewidth=4)
    # ax2.legend(fontsize=16)
    ax2.set_xlabel('Time (s)', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Wheel Position ($\\theta$) (rad)', fontsize=18, fontweight='bold')
    ax2.set_title('Wheel Position v.s. Time', fontsize=20, fontweight='bold')
    ax2.grid(True)
    ax2.set_xlim(-0.1, time_array[-1])
    ax2.tick_params(labelsize=16)
    
    # Plot grip forces
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111)
    ax3.plot(time_array, avg_grip_light, 'k-', label='Light', linewidth=4)
    ax3.plot(time_array, avg_grip_heavy, 'b-', label='Heavy', linewidth=4)
    # ax3.legend(fontsize=16)
    ax3.set_xlabel('Time (s)', fontsize=18, fontweight='bold')
    ax3.set_ylabel('Grip Force (N)', fontsize=18, fontweight='bold')
    ax3.set_title('Grip Forces v.s. Time', fontsize=20, fontweight='bold')
    ax3.grid(True)
    ax3.set_xlim(-0.1, time_array[-1])
    ax3.tick_params(labelsize=16)
    
    # Next, we want to find a catch trial the virtual trajectories' prediction against the actual trajectory. 
    # To get the virtual trajectory prediction, we need to integrate the differential equation: 
    # (phi - theta)k + (phi_dot - theta_dot)b - CommandTorque = J*theta_ddot
    # We can do this using scipy's solve_ivp function.
    def theta_ddot_light_heavy(t, y): 
        theta, theta_dot = y 
        grip = avg_grip_light[np.argmin(np.abs(time_array - t))]
        j_light = j1 * grip + j0
        b_light = b1 * grip + b0
        k_light = k1 * grip + k0
        command_torque = np.clip(theta*(0.4/0.1), 0, 0.4)
        forces = (phi_light[np.argmin(np.abs(time_array - t))] - theta)*k_light + (phi_light_dot[np.argmin(np.abs(time_array - t))] - theta_dot)*b_light - command_torque
        theta_ddot = np.sum(forces)/j_light
        
        return [theta_dot, theta_ddot]
    
    theta_light_heavy = solve_ivp(theta_ddot_light_heavy, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
    
    def theta_ddot_heavy_light(t, y): 
        theta, theta_dot = y 
        grip = avg_grip_heavy[np.argmin(np.abs(time_array - t))]
        j_heavy = j1 * grip + j0
        b_heavy = b1 * grip + b0
        k_heavy = k1 * grip + k0
        command_torque = np.clip(theta*(0.1/0.1), 0, 0.1)
        forces = (phi_heavy[np.argmin(np.abs(time_array - t))] - theta)*k_heavy + (phi_heavy_dot[np.argmin(np.abs(time_array - t))] - theta_dot)*b_heavy - command_torque
        theta_ddot = np.sum(forces)/j_heavy
        
        return [theta_dot, theta_ddot]
    
    theta_heavy_light = solve_ivp(theta_ddot_heavy_light, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
    
    # Print the shapes of phi_light and phi_heavy, avg_pos_heavy_light and theta_heavy_light, and theta_light_heavy
    print(f"phi_light shape: {phi_light.shape}, phi_heavy shape: {phi_heavy.shape}")
    print(f"avg_pos_heavy_light shape: {avg_pos_heavy_light.shape}, theta_heavy_light shape: {theta_heavy_light.y[0].shape}")
    print(f"theta_light_heavy shape: {theta_light_heavy.y[0].shape}")
    
    # Plot Heavy->Light catch trial
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111)
    ax4.plot(time_array, avg_pos_heavy_light, 'r-', label='Experimental Data', linewidth=6)
    ax4.plot(theta_heavy_light.t, theta_heavy_light.y[0], 'r:', label='Model Prediction', linewidth=6)
    ax4.plot(time_array, phi_heavy, 'b:', label='Virtual Trajectory', linewidth=6)
    ax4.legend(fontsize=16)
    ax4.set_xlabel('Time (s)', fontsize=18, fontweight='bold')
    ax4.set_ylabel('Point Position ($\\theta$) (rad)', fontsize=18, fontweight='bold')
    ax4.set_title('Input Estimation and Prediction of Heavy->Light Catch Trial', fontsize=20, fontweight='bold')
    ax4.grid(True)
    ax4.tick_params(labelsize=16)
    
    # Plot Light->Heavy catch trial
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111)
    ax5.plot(time_array, avg_pos_light_heavy, 'g-', label='Experimental Data', linewidth=6)
    ax5.plot(theta_light_heavy.t, theta_light_heavy.y[0], 'g:', label='Model Prediction', linewidth=6)
    ax5.plot(time_array, phi_light, 'k:', label='Virtual Trajectory', linewidth=6)
    ax5.legend(fontsize=16)
    ax5.set_xlabel('Time (s)', fontsize=18, fontweight='bold')
    ax5.set_ylabel('Position ($\\theta$) (rad)', fontsize=18, fontweight='bold')
    ax5.set_title('Input Estimation and Prediction of Light->Heavy Catch Trial', fontsize=20, fontweight='bold')
    ax5.grid(True)
    ax5.tick_params(labelsize=16)
    
    # Calculate VAF for the catch trials up to 0.1s To do this, we need to find the index of the 0.1s mark 
    # in the time array, and then calculate the VAF between the experimental data and the model prediction. 
    # Find index corresponding to 0.1s
    idx_100ms = np.argmin(np.abs(time_array - 0.1))
    vaf_heavy_light = 1 - np.sum(np.abs(avg_pos_heavy_light[:idx_100ms] - theta_heavy_light.y[0][:idx_100ms])**2)/np.sum(np.abs(avg_pos_heavy_light[:idx_100ms])**2)
    vaf_light_heavy = 1 - np.sum(np.abs(avg_pos_light_heavy[:idx_100ms] - theta_light_heavy.y[0][:idx_100ms])**2)/np.sum(np.abs(avg_pos_light_heavy[:idx_100ms])**2)
    
    print(f"VAF Heavy->Light: {vaf_heavy_light*100:.2f}%, VAF Light->Heavy: {vaf_light_heavy*100:.2f}%")
    
    # Save figures if requested
    if save_fig and save_path:
        # Create a directory for the figures if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save each figure with a unique name
        if output_dir:
            fig1.savefig(os.path.join(output_dir, 'virtual_trajectories_control_input.png'))
            fig2.savefig(os.path.join(output_dir, 'virtual_trajectories_wheel_position.png'))
            fig3.savefig(os.path.join(output_dir, 'virtual_trajectories_grip_force.png'))
            fig4.savefig(os.path.join(output_dir, 'virtual_trajectories_heavy_light_catch.png'))
            fig5.savefig(os.path.join(output_dir, 'virtual_trajectories_light_heavy_catch.png'))
        else:
            # If no output_dir is provided, use the save_path as a base and add suffixes
            base_path = os.path.splitext(save_path)[0]
            fig1.savefig(f"{base_path}_control_input.png")
            fig2.savefig(f"{base_path}_wheel_position.png")
            fig3.savefig(f"{base_path}_grip_force.png")
            fig4.savefig(f"{base_path}_heavy_light_catch.png")
            fig5.savefig(f"{base_path}_light_heavy_catch.png")
    
    plt.show()

    
def plot_virtual_trajectory_results_for_paper(results, output_dir=None): 
    """
    Plot results of simulating the virtual trajectory calculated with: 
    - Individualized impedance parameters 
    - Group functional impedance parameters 
    - Group constant impedance parmaeters
    
    We use the VAF metric, and we compare for both heavy-light and light-heavy catch trials. 
    
    """
    
    # To do this, first we need to get the impedance v.s. grip force threshold data, as calculated in. 
    # plot_calibration_results_for_paper. 
    # Then, we need to get the main trials by load case as in plot_main_trial_position_and_grip. 
    # Then, we can plot the virtual trajectories for each load case.
    
    print(f"Plotting virtual trajectory results for {len(results)} participants")
    
    vaf_light_heavy_individualized = [] 
    vaf_heavy_light_individualized = [] 
    vaf_light_heavy_functional = [] 
    vaf_heavy_light_functional = [] 
    vaf_light_heavy_constant = [] 
    vaf_heavy_light_constant = [] 

    j1_all = 1.9e-5
    j0_all = 3.3e-4
    b1_all = 1.2e-3
    b0_all = 2.7e-2
    k1_all = 0.067
    k0_all = 0.91
    
    for participant in results.keys(): 
        
        print(f"Processing participant {participant}")

        # Get calibration data and calculate impedance parameters vs grip force 
        calibration_runs = results[participant]['calibration']
        single_participant_df = pd.DataFrame()
        
        # Collect valid calibration trials 
        for run_idx, run in calibration_runs.items(): 
            metrics = [] 
            for trial in run['trials'].values(): 
                if trial['is_valid'] and trial['metrics']['grip_force_threshold'] < 30: 
                    metrics.append({
                        'inertia': trial['metrics']['inertia'], 
                        'damping': trial['metrics']['damping'], 
                        'stiffness': trial['metrics']['stiffness'], 
                        'grip_force_threshold': trial['metrics']['grip_force_threshold']
                    })
            run_df = pd.DataFrame(metrics)
            single_participant_df = pd.concat([single_participant_df, run_df], ignore_index=True)
            
        # Calculate the mean parameters for each grip force threshold
        mean_results_df = single_participant_df.groupby('grip_force_threshold').mean().reset_index()
        
        # Calculate linear fits for impedance parameters
        grip = mean_results_df['grip_force_threshold']
        j1, j0 = np.polyfit(grip, mean_results_df['inertia'], 1)
        b1, b0 = np.polyfit(grip, mean_results_df['damping'], 1)
        k1, k0 = np.polyfit(grip, mean_results_df['stiffness'], 1)
        
        # Now get the main trial runs 
        main_runs = results[participant]['main']
        
        trial_types = split_runs_into_trial_types(main_runs)
        light_trials = trial_types['light']
        heavy_trials = trial_types['heavy']
        light_heavy_trials = trial_types['light_heavy']
        heavy_light_trials = trial_types['heavy_light']
        
        print(f"Light trials: {len(light_trials)}, Heavy trials: {len(heavy_trials)}, Light->Heavy trials: {len(light_heavy_trials)}, Heavy->Light trials: {len(heavy_light_trials)}")
    
        first_trial = None
        for trial_list in [light_trials, heavy_trials, light_heavy_trials, heavy_light_trials]:
            if trial_list:
                first_trial = trial_list[0]
                break
                
        if first_trial is None:
            print("No valid trials found")
            continue
        
        # Find index nearest to 0.3s for the first trial
        shared_final_index = np.argmin(np.abs(first_trial['elapsed_time'] - 0.3))
        time_array = first_trial['elapsed_time'].iloc[0:shared_final_index].values
        
        
        # Initialize arrays to store average trajectories
        avg_pos_light = np.zeros_like(time_array)
        avg_vel_light = np.zeros_like(time_array)
        avg_accel_light = np.zeros_like(time_array)
        avg_pos_heavy = np.zeros_like(time_array)
        avg_vel_heavy = np.zeros_like(time_array)
        avg_accel_heavy = np.zeros_like(time_array)
        avg_grip_light = np.zeros_like(time_array)
        avg_grip_heavy = np.zeros_like(time_array)
        avg_torque_light = np.zeros_like(time_array)
        avg_torque_heavy = np.zeros_like(time_array)
        avg_pos_light_heavy = np.zeros_like(time_array)
        avg_pos_heavy_light = np.zeros_like(time_array)
        avg_grip_light_heavy = np.zeros_like(time_array)
        avg_grip_heavy_light = np.zeros_like(time_array)
        
        # Count number of valid trials for each case
        n_light = 0
        n_heavy = 0
        n_light_heavy = 0
        n_heavy_light = 0
        
        # Sum up trajectories for light trials
        for trial in light_trials:
            pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
            vel = trial['EncoderRadians_dot_smooth'].iloc[0:shared_final_index].values
            accel = trial['EncoderRadians_ddot_smooth'].iloc[0:shared_final_index].values
            grip = trial['GripForce'].iloc[0:shared_final_index].values
            torque = trial['CommandTorque'].iloc[0:shared_final_index].values
            avg_pos_light += pos
            avg_vel_light += vel
            avg_accel_light += accel
            avg_grip_light += grip
            avg_torque_light += torque
            n_light += 1
            
        # Sum up trajectories for heavy trials  
        for trial in heavy_trials:
            pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
            vel = trial['EncoderRadians_dot_smooth'].iloc[0:shared_final_index].values
            accel = trial['EncoderRadians_ddot_smooth'].iloc[0:shared_final_index].values
            grip = trial['GripForce'].iloc[0:shared_final_index].values
            torque = trial['CommandTorque'].iloc[0:shared_final_index].values
            avg_pos_heavy += pos
            avg_vel_heavy += vel
            avg_accel_heavy += accel
            avg_grip_heavy += grip
            avg_torque_heavy += torque
            n_heavy += 1
            
        for trial in light_heavy_trials:
            pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
            grip = trial['GripForce'].iloc[0:shared_final_index].values
            avg_pos_light_heavy += pos
            avg_grip_light_heavy += grip
            n_light_heavy += 1
            
        for trial in heavy_light_trials:
            pos = trial['EncoderRadians_smooth'].iloc[0:shared_final_index].values
            grip = trial['GripForce'].iloc[0:shared_final_index].values
            avg_pos_heavy_light += pos
            avg_grip_heavy_light += grip
            n_heavy_light += 1
            
        # Calculate averages
        if n_light > 0:
            avg_pos_light /= n_light
            avg_vel_light /= n_light
            avg_accel_light /= n_light
            avg_grip_light /= n_light
            avg_torque_light /= n_light
        
        if n_heavy > 0:
            avg_pos_heavy /= n_heavy
            avg_vel_heavy /= n_heavy
            avg_accel_heavy /= n_heavy
            avg_grip_heavy /= n_heavy
            avg_torque_heavy /= n_heavy
        
        if n_light_heavy > 0:
            avg_pos_light_heavy /= n_light_heavy
            avg_grip_light_heavy /= n_light_heavy
            
        if n_heavy_light > 0:
            avg_pos_heavy_light /= n_heavy_light
            avg_grip_heavy_light /= n_heavy_light
            
            
            
        def phi_dot_light(t, y, impedance_params, constant_impedance = None): 
            theta = avg_pos_light[np.argmin(np.abs(time_array - t))]
            theta_dot = avg_vel_light[np.argmin(np.abs(time_array - t))]
            theta_ddot = avg_accel_light[np.argmin(np.abs(time_array - t))]
            grip = avg_grip_light[np.argmin(np.abs(time_array - t))]
            command_torque = np.clip(theta*(0.1/0.1), 0, 0.1)
            
            j1 = impedance_params['j1']
            j0 = impedance_params['j0']
            b1 = impedance_params['b1']
            b0 = impedance_params['b0']
            k1 = impedance_params['k1']
            k0 = impedance_params['k0']
            
            if constant_impedance is not None: 
                j_light = constant_impedance['j']
                b_light = constant_impedance['b']
                k_light = constant_impedance['k']
            else: 
                j_light = j1 * grip + j0
                b_light = b1 * grip + b0
                k_light = k1 * grip + k0
                
            return -(k_light/b_light)*y[0] + (1/b_light)*(k_light*theta + b_light*theta_dot + j_light*theta_ddot + command_torque)

        def phi_dot_heavy(t, y, impedance_params, constant_impedance = None): 
            theta = avg_pos_heavy[np.argmin(np.abs(time_array - t))]
            theta_dot = avg_vel_heavy[np.argmin(np.abs(time_array - t))]
            theta_ddot = avg_accel_heavy[np.argmin(np.abs(time_array - t))]
            grip = avg_grip_heavy[np.argmin(np.abs(time_array - t))]
            command_torque = np.clip(theta*(0.4/0.1), 0, 0.4)
            
            j1 = impedance_params['j1']
            j0 = impedance_params['j0']
            b1 = impedance_params['b1']
            b0 = impedance_params['b0']
            k1 = impedance_params['k1']
            k0 = impedance_params['k0']
            
            if constant_impedance is not None: 
                j_heavy = constant_impedance['j']
                b_heavy = constant_impedance['b']
                k_heavy = constant_impedance['k']
            else: 
                j_heavy = j1 * grip + j0
                b_heavy = b1 * grip + b0
                k_heavy = k1 * grip + k0
                
            return -(k_heavy/b_heavy)*y[0] + (1/b_heavy)*(k_heavy*theta + b_heavy*theta_dot + j_heavy*theta_ddot + command_torque)
    

        impedance_params = {
            'j1': j1, 
            'j0': j0, 
            'b1': b1, 
            'b0': b0, 
            'k1': k1, 
            'k0': k0
        }
        phi_dot_light_individualized = partial(phi_dot_light, impedance_params=impedance_params)
        phi_dot_heavy_individualized = partial(phi_dot_heavy, impedance_params=impedance_params)
        
        impedance_params_functional = {
            'j1': j1_all, 
            'j0': j0_all, 
            'b1': b1_all, 
            'b0': b0_all, 
            'k1': k1_all, 
            'k0': k0_all
        }
        phi_dot_light_functional = partial(phi_dot_light, impedance_params=impedance_params_functional)
        phi_dot_heavy_functional = partial(phi_dot_heavy, impedance_params=impedance_params_functional)
        
        # Calculate average impedance parameters for constant impedance. Get the average 
        # of the three linear fits across the grip range 5-30
        j_avg = ((j1_all * 30 + j0_all) + (j1_all * 5 + j0_all))/2
        b_avg = ((b1_all * 30 + b0_all) + (b1_all * 5 + b0_all))/2
        k_avg = ((k1_all * 30 + k0_all) + (k1_all * 5 + k0_all))/2
        
        impedance_params_constant = {
            'j': j_avg, 
            'b': b_avg, 
            'k': k_avg
        }
        phi_dot_light_constant = partial(phi_dot_light, impedance_params=impedance_params_functional, constant_impedance=impedance_params_constant)
        phi_dot_heavy_constant = partial(phi_dot_heavy, impedance_params=impedance_params_functional, constant_impedance=impedance_params_constant)
        
        
        phi_light_individualized = solve_ivp(phi_dot_light_individualized, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        phi_heavy_individualized = solve_ivp(phi_dot_heavy_individualized, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        
        phi_light_functional = solve_ivp(phi_dot_light_functional, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        phi_heavy_functional = solve_ivp(phi_dot_heavy_functional, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        
        phi_light_constant = solve_ivp(phi_dot_light_constant, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        phi_heavy_constant = solve_ivp(phi_dot_heavy_constant, [0, time_array[-1]], [0], t_eval=time_array, method='RK45')
        
        phi_light_individualized = np.array(phi_light_individualized.y[0])
        phi_light_dot_individualized = np.array([phi_dot_light_individualized(t, [phi_light_individualized[i]]) for i, t in enumerate(time_array)])
        phi_heavy_individualized = np.array(phi_heavy_individualized.y[0])
        phi_heavy_dot_individualized = np.array([phi_dot_heavy_individualized(t, [phi_heavy_individualized[i]]) for i, t in enumerate(time_array)])
        
        phi_light_functional = np.array(phi_light_functional.y[0])
        phi_light_dot_functional = np.array([phi_dot_light_functional(t, [phi_light_functional[i]]) for i, t in enumerate(time_array)])
        phi_heavy_functional = np.array(phi_heavy_functional.y[0])
        phi_heavy_dot_functional = np.array([phi_dot_heavy_functional(t, [phi_heavy_functional[i]]) for i, t in enumerate(time_array)])
        
        phi_light_constant = np.array(phi_light_constant.y[0])
        phi_light_dot_constant = np.array([phi_dot_light_constant(t, [phi_light_constant[i]]) for i, t in enumerate(time_array)])
        phi_heavy_constant = np.array(phi_heavy_constant.y[0])
        phi_heavy_dot_constant = np.array([phi_dot_heavy_constant(t, [phi_heavy_constant[i]]) for i, t in enumerate(time_array)])
        
        
        def theta_ddot_light_heavy(t, y, phi, phi_dot, impedance_params, constant_impedance = None): 
            theta, theta_dot = y 
            grip = avg_grip_light[np.argmin(np.abs(time_array - t))]
            
            if constant_impedance is not None: 
                j_light = constant_impedance['j']
                b_light = constant_impedance['b']
                k_light = constant_impedance['k']
            else: 
                j_light = j1 * grip + j0
                b_light = b1 * grip + b0
                k_light = k1 * grip + k0
                
            command_torque = np.clip(theta*(0.4/0.1), 0, 0.4)
            forces = (phi[np.argmin(np.abs(time_array - t))] - theta)*k_light + (phi_dot[np.argmin(np.abs(time_array - t))] - theta_dot)*b_light - command_torque
            theta_ddot = np.sum(forces)/j_light
            
            return [theta_dot, theta_ddot]
            
            
        def theta_ddot_heavy_light(t, y, phi, phi_dot, impedance_params, constant_impedance = None): 
            theta, theta_dot = y 
            grip = avg_grip_heavy[np.argmin(np.abs(time_array - t))]
            
            if constant_impedance is not None: 
                j_heavy = constant_impedance['j']
                b_heavy = constant_impedance['b']
                k_heavy = constant_impedance['k']
            else: 
                j_heavy = j1 * grip + j0
                b_heavy = b1 * grip + b0
                k_heavy = k1 * grip + k0
                
            command_torque = np.clip(theta*(0.1/0.1), 0, 0.1)
            forces = (phi[np.argmin(np.abs(time_array - t))] - theta)*k_heavy + (phi_dot[np.argmin(np.abs(time_array - t))] - theta_dot)*b_heavy - command_torque
            theta_ddot = np.sum(forces)/j_heavy
            
            return [theta_dot, theta_ddot]
        
        theta_ddot_light_heavy_individualized = partial(theta_ddot_light_heavy, phi=phi_light_individualized, phi_dot=phi_light_dot_individualized, impedance_params=impedance_params)
        theta_ddot_heavy_light_individualized = partial(theta_ddot_heavy_light, phi=phi_heavy_individualized, phi_dot=phi_heavy_dot_individualized, impedance_params=impedance_params)
        
        theta_ddot_light_heavy_functional = partial(theta_ddot_light_heavy, phi=phi_light_functional, phi_dot=phi_light_dot_functional, impedance_params=impedance_params_functional)
        theta_ddot_heavy_light_functional = partial(theta_ddot_heavy_light, phi=phi_heavy_functional, phi_dot=phi_heavy_dot_functional, impedance_params=impedance_params_functional)
        
        theta_ddot_light_heavy_constant = partial(theta_ddot_light_heavy, phi=phi_light_constant, phi_dot=phi_light_dot_constant, impedance_params=impedance_params_functional, constant_impedance=impedance_params_constant)
        theta_ddot_heavy_light_constant = partial(theta_ddot_heavy_light, phi=phi_heavy_constant, phi_dot=phi_heavy_dot_constant, impedance_params=impedance_params_functional, constant_impedance=impedance_params_constant)
    
        theta_light_heavy_individualized = solve_ivp(theta_ddot_light_heavy_individualized, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
        theta_heavy_light_individualized = solve_ivp(theta_ddot_heavy_light_individualized, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
        
        theta_light_heavy_functional = solve_ivp(theta_ddot_light_heavy_functional, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
        theta_heavy_light_functional = solve_ivp(theta_ddot_heavy_light_functional, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
        
        theta_light_heavy_constant = solve_ivp(theta_ddot_light_heavy_constant, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
        theta_heavy_light_constant = solve_ivp(theta_ddot_heavy_light_constant, [0, time_array[-1]], [0, 0], t_eval=time_array, method='RK45')
    
    
        # Turn into numpy arrays
        theta_light_heavy_individualized = np.array(theta_light_heavy_individualized.y[0])
        theta_heavy_light_individualized = np.array(theta_heavy_light_individualized.y[0])
        theta_light_heavy_functional = np.array(theta_light_heavy_functional.y[0])
        theta_heavy_light_functional = np.array(theta_heavy_light_functional.y[0])
        theta_light_heavy_constant = np.array(theta_light_heavy_constant.y[0])
        theta_heavy_light_constant = np.array(theta_heavy_light_constant.y[0])
    
        idx_100ms = np.argmin(np.abs(time_array - 0.1))
        
        vaf_light_heavy_individualized_i = 1 - np.sum(np.abs(avg_pos_light_heavy[:idx_100ms] - theta_light_heavy_individualized[:idx_100ms])**2)/np.sum(np.abs(avg_pos_light_heavy[:idx_100ms])**2)
        vaf_heavy_light_individualized_i = 1 - np.sum(np.abs(avg_pos_heavy_light[:idx_100ms] - theta_heavy_light_individualized[:idx_100ms])**2)/np.sum(np.abs(avg_pos_heavy_light[:idx_100ms])**2)

        vaf_light_heavy_individualized.append(vaf_light_heavy_individualized_i)
        vaf_heavy_light_individualized.append(vaf_heavy_light_individualized_i)
        
        vaf_light_heavy_functional_i = 1 - np.sum(np.abs(avg_pos_light_heavy[:idx_100ms] - theta_light_heavy_functional[:idx_100ms])**2)/np.sum(np.abs(avg_pos_light_heavy[:idx_100ms])**2)
        vaf_heavy_light_functional_i = 1 - np.sum(np.abs(avg_pos_heavy_light[:idx_100ms] - theta_heavy_light_functional[:idx_100ms])**2)/np.sum(np.abs(avg_pos_heavy_light[:idx_100ms])**2)

        vaf_light_heavy_functional.append(vaf_light_heavy_functional_i)
        vaf_heavy_light_functional.append(vaf_heavy_light_functional_i)
        
        vaf_light_heavy_constant_i = 1 - np.sum(np.abs(avg_pos_light_heavy[:idx_100ms] - theta_light_heavy_constant[:idx_100ms])**2)/np.sum(np.abs(avg_pos_light_heavy[:idx_100ms])**2)
        vaf_heavy_light_constant_i = 1 - np.sum(np.abs(avg_pos_heavy_light[:idx_100ms] - theta_heavy_light_constant[:idx_100ms])**2)/np.sum(np.abs(avg_pos_heavy_light[:idx_100ms])**2)

        vaf_light_heavy_constant.append(vaf_light_heavy_constant_i)
        vaf_heavy_light_constant.append(vaf_heavy_light_constant_i)

   
    # Once this is all done, create a dataframe with VAFs for each participant, and columns for individualized, functional, and constant
    vaf_df = pd.DataFrame({
        'participant': results.keys(),
        'vaf_light_heavy_individualized': vaf_light_heavy_individualized,
        'vaf_heavy_light_individualized': vaf_heavy_light_individualized,
        'vaf_light_heavy_functional': vaf_light_heavy_functional,
        'vaf_heavy_light_functional': vaf_heavy_light_functional,
        'vaf_light_heavy_constant': vaf_light_heavy_constant,
        'vaf_heavy_light_constant': vaf_heavy_light_constant
    })
    
    # Now we want to make a bar plot of the VAFs for the group. Additionally exclude elements that are -inf or nan
    vaf_light_heavy_individualized_avg = np.mean([vaf for vaf in vaf_df['vaf_light_heavy_individualized'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    vaf_heavy_light_individualized_avg = np.mean([vaf for vaf in vaf_df['vaf_heavy_light_individualized'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    vaf_light_heavy_functional_avg = np.mean([vaf for vaf in vaf_df['vaf_light_heavy_functional'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    vaf_heavy_light_functional_avg = np.mean([vaf for vaf in vaf_df['vaf_heavy_light_functional'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    vaf_light_heavy_constant_avg = np.mean([vaf for vaf in vaf_df['vaf_light_heavy_constant'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    vaf_heavy_light_constant_avg = np.mean([vaf for vaf in vaf_df['vaf_heavy_light_constant'] if vaf != -np.inf and vaf != np.nan and vaf > -1])
    
    # The bar chart should be 3 groups of 2 bars each. The first group should be the individualized, the second group should be the functional, and the third group should be the constant
    # The bar groups should be labeled as such and the bars within each group should be labeled 
    # iether light->heavy or heavy->light
    
    # Create figure and axis
    fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
    
    # Set width of bars and positions of the bars
    width = 0.35
    x = np.arange(3)
    
    # Create bars
    rects1 = ax.bar(x - width/2, [vaf_light_heavy_individualized_avg*100, 
                                 vaf_light_heavy_functional_avg*100,
                                 vaf_light_heavy_constant_avg*100], 
                    width, label='Light->Heavy', color='green', linewidth=2)
    
    rects2 = ax.bar(x + width/2, [vaf_heavy_light_individualized_avg*100,
                                 vaf_heavy_light_functional_avg*100, 
                                 vaf_heavy_light_constant_avg*100],
                    width, label='Heavy->Light', color='red', linewidth=2)

    # Customize plot
    ax.set_ylabel('VAF (%)', fontsize=18, fontweight='bold')
    ax.set_title('Variance Accounted For (VAF) by Model Type (with potential outliers removed)', fontsize=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Individualized', 'Functional', 'Constant'], fontsize=16, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=14,
                       fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    
    print(f"VAFs Results:") 
    print(vaf_df)
    
    plt.show()
    
