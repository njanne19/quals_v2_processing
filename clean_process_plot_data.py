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
    plot_average_trajectories_single_participant
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
        'dataset_name' : 'mar27_02',
        'participants' : [
            'Mark_Mar27', 
            'Hannah_B_Mar27', 
            'Nick_Mar26_TestProc',
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

        plot_main_load_schedule(results[participant]['main']['runs'])
        # plot_single_trial(results[participant]['main']['runs'][54])
        estimate_average_trajectories_single_participant(results[participant], 'main')
        plot_average_trajectories_single_participant(results[participant])
        plot_trial_list(results[participant]['main']['runs'], title=f'{participant} Main Runs', show_outliers=True, just_outliers=True)
        plt.show()
            
        

    return 
        
    task_list = [] 
    
    for trial_participant_dir in trial_participant_dirs: 
        base_name = os.path.basename(trial_participant_dir) 
        results[base_name] = {
            'basepath': trial_participant_dir, 
        }
        
        calibration_files = glob.glob(os.path.join(trial_participant_dir, 'calibration', '*.csv'))
        results[base_name]['num_calibration_runs'] = len(calibration_files)
        results[base_name]['calibration_files'] = calibration_files
        results[base_name]['calibration'] = {}

        main_files = glob.glob(os.path.join(trial_participant_dir, 'main', '*.csv'))
        results[base_name]['num_main_runs'] = len(main_files)
        results[base_name]['main_files'] = main_files
        results[base_name]['main'] = {}
        # Now we want to create a task list for each of the calibration files 
        for idx, file in enumerate(calibration_files): 
            task_list.append((base_name, 'calibration', file, idx))

        # Now we want to create a task list for each of the main files 
        for idx, file in enumerate(main_files): 
            task_list.append((base_name,'main', file, idx))
    
   
    # Then we iterate over the task list and process the data in parallel 
    # the task list partially reveals where the data should go in the results directory 
    # use multiprocessing to process the data in parallel
    print(f"Processing {len(task_list)} tasks")
    with multiprocessing.Pool(os.cpu_count()-1) as pool:
        with tqdm(total=len(task_list)) as pbar:
            for (base_name, task_type, idx), value in pool.imap_unordered(execute_processing_task, task_list):
                results[base_name][task_type][idx] = value
                pbar.update()

    print(f"Completed processing {len(task_list)} tasks")
                
    # Then from this list of final results, 
    # plot me a random calibration run. Show the file where it came from, 
    # all trials included in the calibration run. 
    # Pick a random participant and calibration run to plot
    # plot_calibration_results_for_paper(results)
    
    # Choose a random participant and plot all their main trials 
    random_participant = np.random.choice(list(results.keys()))
    # plot_main_trial_position_and_grip(results, random_participant)
    # plot_calibration_results_and_trials(results, random_participant)
    # plot_calibration_results_for_paper(results)
    # plot_main_trial_position_and_grip(results, 'Dev-Nov21')
    # plot_virtual_trajectories_for_paper(results, 'Dev-Nov21')
    plot_virtual_trajectory_results_for_paper(results)
    # for participant in results.keys():
        # plot_virtual_trajectories_for_paper(results, participant)
        # plot_calibration_results_and_trials(results, participant)
    # plot_virtual_trajectories_for_paper(results, 'Dev-Nov21')
    # plt.show()
    
    # if results[random_participant]['main']:
    #     num_runs = len(results[random_participant]['main'])
        
    #     # Create figure with 5 rows (schedule + 4 time series) and num_runs columns
    #     fig, axes = plt.subplots(5, num_runs, figsize=(6*num_runs, 12), height_ratios=[1, 1, 1, 1, 1])
    #     if num_runs == 1:
    #         axes = axes.reshape(-1, 1)
    #     fig.suptitle(f'Main Trial Analysis\nParticipant: {random_participant}')
        
    #     # Process each run in its own column
    #     for run_idx in range(num_runs):
    #         main_data = results[random_participant]['main'][run_idx]
            
    #         # Create schedule plot
    #         load_map_numeric = {
    #             'very_light': 1,
    #             'light': 2, 
    #             'medium': 3,
    #             'moderately_heavy': 4,
    #             'heavy': 5
    #         }
            
    #         unique_loads = list(set(trial['load'] for trial in main_data['trials'].values()))
    #         colors = plt.cm.rainbow(np.linspace(0, 1, len(main_data['trials'])))
            
    #         # Plot schedule
    #         for trial_idx, trial_data in main_data['trials'].items():
    #             load = trial_data['load']
    #             load_level = load_map_numeric[load]
    #             marker = 'x' if not trial_data['is_valid'] else 'o'
    #             axes[0,run_idx].scatter(trial_idx, load_level, color=colors[trial_idx], s=100, marker=marker)
                
    #         axes[0,run_idx].set_yticks(range(1,6))
    #         axes[0,run_idx].set_yticklabels(['Very Light', 'Light', 'Medium', 'Moderately Heavy', 'Heavy'])
    #         axes[0,run_idx].set_xlabel('Trial Number')
    #         axes[0,run_idx].set_title(f'Run {run_idx} Schedule')
            
    #         # Plot time series for all trials
    #         for trial_idx, trial_data in main_data['trials'].items():
    #             df = trial_data['data']
    #             time = df['elapsed_time']
    #             color = colors[trial_idx]
    #             linestyle = '--' if not trial_data['is_valid'] else '-'
                
    #             # Plot encoder position
    #             axes[1,run_idx].plot(time, df['EncoderRadians'], c=color, alpha=0.7, linestyle=linestyle)
    #             axes[1,run_idx].set_ylabel('Position (rad)')
                
    #             # Plot encoder velocity  
    #             axes[2,run_idx].plot(time, df['EncoderRadians_dot_smooth'], c=color, alpha=0.7, linestyle=linestyle)
    #             axes[2,run_idx].set_ylabel('Velocity (rad/s)')
                
    #             # Plot grip force
    #             axes[3,run_idx].plot(time, df['GripForce'], c=color, alpha=0.7, linestyle=linestyle)
    #             axes[3,run_idx].set_ylabel('Grip Force (N)')
                
    #             # Plot command torque
    #             axes[4,run_idx].plot(time, df['CommandTorque'], c=color, alpha=0.7, linestyle=linestyle)
    #             axes[4,run_idx].set_ylabel('Command Torque (Nm)')
    #             axes[4,run_idx].set_xlabel('Time (s)')
            
    #         # Add grid to all plots in this column
    #         for ax in axes[:,run_idx]:
    #             ax.grid(True, alpha=0.3)
                
    #     plt.tight_layout()
    #     plt.show()

    

if __name__ == '__main__': 
    main()