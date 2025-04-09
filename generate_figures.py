import numpy as np 
import pandas as pd 
import yaml 
import matplotlib.pyplot as plt 
import glob
import os 
from processing import (
    read_and_preprocess_data, 
    calculate_calibration_metrics_single_trial, 
    find_live_indices
)
from plotting import plot_single_trial, plot_combined_trials, plot_calibration_metrics, plot_aggregate_calibration_metrics

def main(mode = ['main']): 
    # Load config 
    with open('config.yml', 'r') as f: 
        config = yaml.safe_load(f)


    if 'calibration' in mode: 
        # Search the calibration data directory for all calibration csv files
        calibration_files = glob.glob('./max_data/*.csv')
        # calibration_files = ['./calibration_data/brent1.csv']
        aggregate_metrics = []

        # For each file, read in the data, preprocess it, and calculate the calibration metrics
        for file in calibration_files: 
            print(f"Processing file: {file}")
            data = read_and_preprocess_data(file)
            
            # First check to see if the results/file directory already exists, if not, create it 
            file_name = os.path.basename(file)
            file_name = os.path.splitext(file_name)[0]
            results_dir = os.path.join('./results', file_name)
            if not os.path.exists(results_dir): 
                os.makedirs(results_dir)
                
            # Then, we want to plot the data for each trial. The data to be plotted is: 
            # time_s, EncoderRadians_smooth, EncoderRadians_dot_smooth, EncoderRadians_ddot_smooth, AppliedTorque, GripForce
            # Start by creating a directory inside the above called 'time-trials'
            time_trials_dir = os.path.join(results_dir, 'time-trials')
            if not os.path.exists(time_trials_dir): 
                os.makedirs(time_trials_dir)
                
            # Now for each trial, plot the data both combined into one plot, and then split up by trial 
            live_indices = find_live_indices(data)
            
            # Plot the combined trials 
            plot_combined_trials(data, live_indices, time_trials_dir)
            
            # Then plot each trial individually 
            for i, index in enumerate(live_indices): 
                start, end = index 
                
                # Plot the data 
                plot_single_trial(data, start, end, i, time_trials_dir)
                
            # Then calculate aggregate calibration metrics 
            aggregate_dir = os.path.join(results_dir, 'aggregate')
            if not os.path.exists(aggregate_dir): 
                os.makedirs(aggregate_dir)
            metrics = calculate_calibration_metrics_single_trial(data)
            plot_calibration_metrics(metrics, aggregate_dir)
            
            print(metrics)
            metrics.to_csv(os.path.join(aggregate_dir, 'calibration_metrics.csv'), index=False)
            
            # Append to aggregate metrics
            aggregate_metrics.append(metrics) 
            
        # Now we want to plot the aggregate metrics 
        aggregate_metrics_dir = os.path.join('./results', 'calibration_metrics')
        plot_aggregate_calibration_metrics(aggregate_metrics, aggregate_metrics_dir)
        
    if 'main' in mode: 
        
        main_files = glob.glob('./main_data/*.csv')
        
        # For each file, read in the data, preprocess it, and plot time trials
        for file in main_files:
            print(f"Processing file: {file}")
            data = read_and_preprocess_data(file)
            
            # Create results directory structure
            file_name = os.path.basename(file)
            file_name = os.path.splitext(file_name)[0]
            results_dir = os.path.join('./results_main', file_name)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            # Create time-trials directory
            time_trials_dir = os.path.join(results_dir, 'time-trials')
            if not os.path.exists(time_trials_dir):
                os.makedirs(time_trials_dir)
                
            # Find live trial indices
            live_indices = find_live_indices(data)
            
            # Plot combined trials
            plot_combined_trials(data, live_indices, time_trials_dir)
            
            # Plot individual trials
            for i, index in enumerate(live_indices):
                start, end = index
                plot_single_trial(data, start, end, i, time_trials_dir)
    
        
if __name__ == '__main__': 
    main()