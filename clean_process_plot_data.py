import numpy as np 
import pandas as pd 
import yaml 
import json
import matplotlib.pyplot as plt 
import glob 
import os 
from processing import (
    extract_individual_runs, 
    process_individual_calibration_run, 
    process_individual_main_run, 
    detect_calibration_outliers, 
    calculate_complete_calibration_metrics, 
    detect_main_outliers,
    estimate_average_trajectories_single_participant
)
from plotting import (
    plot_single_trial, 
    plot_combined_trials, 
    plot_calibration_metrics, 
    plot_main_trial_position_and_grip, 
    plot_calibration_results_and_trials, 
    plot_calibration_results_for_paper, 
    plot_virtual_trajectories_for_paper, 
    plot_virtual_trajectory_results_for_paper, 
    plot_trial_list, 
    plot_calibration_results_for_batch_and_individual, 
    plot_main_load_schedule, 
    plot_average_trajectories_single_participant, 
    plot_average_trajectories_single_participant_for_paper
)
import multiprocessing
from tqdm import tqdm 
import argparse
from pathlib import Path


def execute_processing_task(task : tuple): 
    base_name, task_type, filepath, idx = task 

    if task_type == 'calibration': 
        data = process_individual_calibration_run(filepath)
    elif task_type == 'main': 
        data = process_individual_main_run(filepath)

    # Now we want to save the data to the results directory 
    return (base_name, task_type, idx), data
    

def main():
    # Set up command line argument parsing
    # Default config if no file specified
    default_config = {
        'dataset_dir': Path(__file__).parent.parent / 'datasets', 
        'dataset_name' : 'apr_15',
        'participants' : [
            # 'Mark_Mar27', 
            # 'Hannah_B_Mar27', 
            # 'Nick_Mar26_TestProc',
            'Tobi_Farbstein_Apr15', 
            'Andrew_Memmer_Apr15',
        ]
    }
    

    parser = argparse.ArgumentParser(description='Process and plot trial data')
    parser.add_argument('--config-file', type=str, default='',
                       help='Config file containing instructions for processing')
    args = parser.parse_args()

    # Load config from file if specified, otherwise use default
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = default_config

    # Read participants.json file to get run indices and other metadata
    participants_file = config['dataset_dir'] / config['dataset_name'] / 'participants.json'
    with open(participants_file, 'r') as f:
        participants_metadata = json.load(f)
    
    # Create a dictionary to store results for each participant
    results = {}
    
    for participant in config['participants']: 
        results[participant] = {}
        
        # To get participant metadata, we need to search by the key "name" in the 
        # "registered_users" list in the participants_metadata dictionary 
        metadata = next(
            (user for user in participants_metadata['registered_users'] 
             if user['name'].strip() == participant.strip()),
            None
        )
        
        if metadata is None:
            print(f"Warning: Could not find metadata for participant {participant}")
            continue
        else: 
            print(f"Processing participant {participant}...")
            
        results[participant]['gender'] = metadata['gender']
        results[participant]['notes'] = metadata['notes']
        results[participant]['basepath'] = config['dataset_dir'] / config['dataset_name'] / participant
    
        results[participant]['calibration'] = {}
    
        # Now we want to get all the files in the calibration directory and sort by date
        calibration_files = glob.glob(os.path.join(results[participant]['basepath'], 'calibration', '*.csv'))
        # Sort files by the datecode in filename (format: ...-YYYY-MM-DD_HH-MM-SS.csv)
        calibration_files.sort(key=lambda x: x.split('-')[-2] + x.split('-')[-1])
        results[participant]['calibration']['num_files'] = len(calibration_files)
        results[participant]['calibration']['files'] = calibration_files
        results[participant]['calibration']['runs'] = []
        results[participant]['calibration']['num_runs'] = 0
        
        # Now we want to process each calibration file
        for file in calibration_files: 
            data = extract_individual_runs(file, 'calibration')
            data = detect_calibration_outliers(data)
            results[participant]['calibration']['runs'].extend(data)
            results[participant]['calibration']['num_runs'] += len(data)
        
        # plot_trial_list(results[participant]['calibration']['runs'], title=f'{participant} Calibration Runs', show_outliers=True)
        calibration_results = calculate_complete_calibration_metrics(results[participant]['calibration']['runs'])
        results[participant]['calibration']['results'] = calibration_results
        print(f"Completed calibration calculations for {participant}") 
        
        # Now we do the main files. We need to align the main files with the "logRecords" element of the metadata file, and attach the metadata
        # to the data from the main files before additional processing. 
        main_files = glob.glob(os.path.join(results[participant]['basepath'], 'main', '*.csv'))
        main_files.sort(key=lambda x: x.split('-')[-2] + x.split('-')[-1])
        results[participant]['main'] = {}
        results[participant]['main']['num_files'] = len(main_files)
        results[participant]['main']['files'] = main_files
        results[participant]['main']['runs'] = []
        results[participant]['main']['num_runs'] = 0
        # results[participant]['main']['logRecords'] = metadata['logRecords']
        
        for file in main_files: 
            data = extract_individual_runs(file, 'main')
            # We look up the log record for this file in the metadata
            # to see if participants were shown the loads or not. 
            for log_record in metadata['logRecords']: 
                if os.path.basename(log_record['filepath']) == os.path.basename(file): 
                    # print(f"Setting show loads to {log_record['showLoads']} for {file}")
                    for run in data: 
                        run.set_show_loads(log_record['showLoads'])
                    break
            results[participant]['main']['runs'].extend(data)
            results[participant]['main']['num_runs'] += len(data)

        # Then we want to iterate through all the runs and set the previous trial load
        loads = {} 
        num_catch_trials = 0 
        num_trials_visible = 0
        for i in range(len(results[participant]['main']['runs'])): 
            if i > 0: 
                results[participant]['main']['runs'][i].set_previous_trial_load(results[participant]['main']['runs'][i-1].get_load())
            else: 
                results[participant]['main']['runs'][i].set_previous_trial_load(None) 
                
            if results[participant]['main']['runs'][i].is_catch_trial(): 
                num_catch_trials += 1
            if results[participant]['main']['runs'][i].is_load_visible(): 
                num_trials_visible += 1
                
            if results[participant]['main']['runs'][i].get_load() not in loads: 
                loads[results[participant]['main']['runs'][i].get_load()] = 0
            loads[results[participant]['main']['runs'][i].get_load()] += 1
            
        # Lastly, now that our trials are labeled we can detect outliers. 
        results[participant]['main']['runs'], num_outliers, num_corrected_outliers = detect_main_outliers(results[participant]['main']['runs'])
            
        print(f"Trial list for {participant}:")
        print(f"Total number of trials: {len(results[participant]['main']['runs'])}")
        print(f"Number of outliers: {num_outliers}")
        print(f"Number of corrected outliers: {num_corrected_outliers}")
        print(f"Number of catch trials: {num_catch_trials}")
        print(f"Number of trials visible: {num_trials_visible}")
        print(f"Loads: {loads}")

        # Create a folder for this participant in the results directory
        participant_results_dir = os.path.join('results', participant)
        if not os.path.exists(participant_results_dir):
            os.makedirs(participant_results_dir)

        # Plot and save calibration results
        fig_calibration_results, _ = plot_calibration_results_for_batch_and_individual(
            calibration_results, 
            results[participant]['main']['runs'], 
            title=f'{participant} Calibration Results',
            save_fig=True,
            save_path=os.path.join(participant_results_dir, 'calibration_results.png')
        )
        plt.close(fig_calibration_results)
            
        # Plot and save main load schedule
        fig_load_schedule, _ = plot_main_load_schedule(
            results[participant]['main']['runs'], 
            title=f'{participant} Main Load Schedule',
            save_fig=True,
            save_path=os.path.join(participant_results_dir, 'main_load_schedule.png')
        )
        plt.close(fig_load_schedule)
        
        
        # Estimate average trajectories
        estimate_average_trajectories_single_participant(results[participant], 'main', use_corrected_outliers=True)
        
        # Estimate virtual trajectories (commented out in original code)
        # estimate_virtual_trajectories_single_participant(results[participant], 'main', use_corrected_outliers=True)
        
        # Plot and save average trajectories for paper
        fig_avg_trajectories = plot_average_trajectories_single_participant_for_paper(
            results[participant],
            save_fig=True,
            save_path=os.path.join(participant_results_dir, 'average_trajectories.png')
        )
        plt.close(fig_avg_trajectories)
        
        # Plot and save trial list
        fig_trial_list, _ = plot_trial_list(
            results[participant]['main']['runs'], 
            title=f'{participant} Main Runs', 
            show_outliers=True, 
            just_outliers=False,
            save_fig=True,
            save_path=os.path.join(participant_results_dir, 'trial_list.png')
        )
        plt.close(fig_trial_list)
            

    return 

    

if __name__ == '__main__': 
    main()