from dash import (
    Dash, 
    html, 
    dcc, 
    callback, 
    Output, 
    Input, 
    dash_table,
    ctx, 
    no_update, 
    ALL
)
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from processing import (
    read_and_preprocess_data, 
    find_live_indices, 
    calculate_calibration_metrics_single_trial
)
import glob
import os

########################################################
#               DATA PROCESSING
########################################################

# Load data. Search for calibration files in ./calibration_data
# Print current directory
print(f"Current directory: {os.getcwd()}")
calibration_files = glob.glob('./calibration_data/*.csv')
main_files = glob.glob('./main_data/*.csv')

print(f"Found {len(calibration_files)} calibration files:")
for file in calibration_files: 
    print(file)

print(f"Found {len(main_files)} main files:")
for file in main_files: 
    print(file)
    
calibration_dfs = [read_and_preprocess_data(file) for file in calibration_files]
main_dfs = [read_and_preprocess_data(file) for file in main_files]

########################################################

########################################################
#               DASH APP, CONFIGURATION
########################################################

# Include tailwind css 
external_scripts = [
    {'src': 'https://cdn.tailwindcss.com'}
]

# Create the dash app and describe the layouts 
app = Dash(__name__, external_scripts=external_scripts) 


app.layout = [
    
    html.Div([
        html.Div([
            html.H1("Human Motor Experiment Dashboard", className='text-4xl font-bold mt-10'),
        ], className="flex justify-center"),
        html.Div([    
            html.H2("Calibration Data Viewer", className='text-2xl font-bold mb-4'),
            dcc.Loading(
                id="loading-experiment-stats",
                children=[
                    html.Div([
                        html.Div([
                            html.H4("Select a data file to view:", className='mb-2'),
                            dcc.Dropdown(
                                id='file-dropdown', 
                                options=[{'label': file, 'value': file} for file in calibration_files], 
                                value=calibration_files[0], 
                                className="max-w-sm"), 
                        ]), 
                        html.Div([ 
                            html.P("Number of Tests Detected:"),
                            html.P('', id='num-experiments-detected', className='font-bold')
                        ]),
                        html.Div([
                            html.P("Total Number of Samples:"), 
                            html.P('', id='total-num-samples', className='font-bold')
                        ]),
                        html.Div([
                            html.P('Total Time Duration (s):'),
                            html.P('', id='total-time-duration', className='font-bold')
                        ]),
                        html.Div([
                            html.P('Average Test Duration (s):'),
                            html.P('', id='average-test-duration', className='font-bold')
                        ]),
                    ], className='grid grid-cols-1 md:grid-cols-5 gap-4'),
                    dcc.Store(id='live-indices-store', storage_type='session', data=[]) 
                ]
            ),
            dcc.Store(id='selected-test-index', data=0), 
            html.Div(id='test-selector-container', className="flex flex-row gap-4 mt-10 overflow-x-auto"),
            html.Div([
                dcc.Loading(
                    id="loading-encoder-radians-plot",
                    type="circle",
                    children=[dcc.Graph(id='encoder-radians-plot')]
                ),
                dcc.Loading(
                    id="loading-encoder-velocity-plot",
                    type="circle", 
                    children=[dcc.Graph(id='encoder-velocity-plot')]
                ),
                dcc.Loading(
                    id="loading-encoder-acceleration-plot",
                    type="circle",
                    children=[dcc.Graph(id='encoder-acceleration-plot')]
                ),
            ], className='grid grid-cols-1 md:grid-cols-3 gap-4'), 
            html.Div([
                dcc.Loading(
                    id="loading-torque-plot",
                    type="circle",
                    children=[dcc.Graph(id='torque-plot')]
                ),
                dcc.Loading(
                    id="loading-grip-force-plot",
                    type="circle",
                    children=[dcc.Graph(id='grip-force-plot')]
                ),
                dcc.Loading(
                    id="loading-test-live-plot",
                    type="circle",
                    children=[dcc.Graph(id='test-live-plot')]
                ),
            ], className='grid grid-cols-1 md:grid-cols-3 gap-4'), 
        ], className="ml-10 mr-10 mt-10"), 
        html.Div([    
            html.H2("Calibration Aggregate Viewer", className='text-2xl font-bold mb-4'),
            html.Div(
                children = [
                html.Div(
                    children = [
                        html.H3("View Configuration", className='text-xl font-bold'), 
                        html.Hr(className="h-px my-2 bg-gray-200 border-0 dark:bg-gray-700"),
                        dcc.Store(id='aggregate-trial-number-store', data=-1),
                        html.Div([
                            html.P("Bulk Views", className='text-lg font-bold'), 
                            
                            # Aggregate-trial-number indicates which trial number is selected for viewing 
                            # -1 indicates all trials are selected. Clcking one of these divs 
                            # should set the aggregate-trial-number-store to the index of the div clicked. 
                            html.Div(
                                children = [
                                    html.P("All Trials", className='text-sm text-gray-500')
                                ],
                                id = {'type': 'aggregate-trial-number', 'index': -1},
                                className="min-h-10 bg-gray-100 rounded-md p-2 hover:bg-gray-200 active:bg-gray-300 transition-colors duration-200 cursor-pointer",
                                style={
                                    'backgroundColor': '#E5E7EB'  # Default gray background
                                }
                            )
                        ]),
                        html.Div([
                            html.P("Individual Views", className="text-lg font-bold"),
                            *[
                                html.Div(
                                    children = [
                                        html.P(file.split('/')[-1].split('.')[0] + f" ({i})", className='text-sm text-gray-500')
                                    ], 
                                    id = {'type': 'aggregate-trial-number', 'index': i},
                                    className="min-h-10 bg-gray-100 rounded-md p-2 hover:bg-gray-200 active:bg-gray-300 transition-colors duration-200 cursor-pointer"
                                )
                                for i, file in enumerate(calibration_files) 
                            ]
                        ])
                        
                    ], 
                    className="min-h-96 col-span-3 md:col-span-1 bg-gray-100 rounded-md p-2"
                ), 
                html.Div(
                    children = [
                        html.H3("Results", className='text-xl font-bold'), 
                        html.Hr(className="h-px my-2 bg-gray-200 border-0 dark:bg-gray-700"),
                        html.Div(id='aggregate-results-container') # Callback will fill this in 
                    ], 
                    className="min-h-96 col-span-3 md:col-span-2 bg-gray-100 rounded-md p-2"
                ), 
                ], 
                className="grid grid-cols-3 gap-4 mb-10"
            )
        ], className="ml-10 mr-10 mt-10"),
    ])
]


@callback(
    Output('aggregate-trial-number-store', 'data'), 
    [Input({'type': 'aggregate-trial-number', 'index': ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def update_aggregate_trial_number_store(*args): 
    """Update the aggregate-trial-number-store when a trial number is clicked."""
    clicked_id = ctx.triggered_id['index']
    return clicked_id

@callback(
    Output({'type': 'aggregate-trial-number', 'index': ALL}, 'style'),
    [Input('aggregate-trial-number-store', 'data')],
    prevent_initial_call=False  # Allow initial call
)
def update_div_styles(active_index):
    """Update the background color of divs based on the active index."""
    print(f"Active index: {active_index}")
    
    # Handle initial None case
    if active_index is None:
        active_index = -1  # Default to "All Trials" selected
        
    # Create a list of styles for each component
    styles = [
        {
            'backgroundColor': '#D1D5DB' if i == active_index else '#F3F4F6',
            'transition': 'background-color 0.2s ease'
        }
        for i in range(-1, len(calibration_files))  # Start from -1 for "All Trials"
    ]
    return styles 

@callback(
    Output('aggregate-results-container', 'children'), 
    [Input('aggregate-trial-number-store', 'data')]
)
def update_aggregate_results_container(active_index): 
    """Update the aggregate results container based on the active index."""
    if active_index == -1: 
        return f"Active index: {active_index}"
    else: 
        
        # Create a simple dash table with fake data         
        df = calibration_dfs[active_index]
        results = calculate_calibration_metrics_single_trial(df)
        
        return dash_table.DataTable(
            results.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},  # Enable horizontal scrolling
        )



@callback(
    [Output('num-experiments-detected', 'children'), 
     Output('total-num-samples', 'children'), 
     Output('total-time-duration', 'children'), 
     Output('average-test-duration', 'children'), 
     Output('live-indices-store', 'data')], 
    [Input('file-dropdown', 'value')]
)
def update_experiment_stats(selected_file): 
    """Updates the number of experiments detected."""
    df = calibration_dfs[calibration_files.index(selected_file)]

    # We need to go through the dataframe and find the independent
    # contiguous sections of indices where the test is live. 
    live_indices = find_live_indices(df)

    total_num_samples = len(df)
    total_time_duration = df['time_s'].iloc[-1]
    average_test_duration = np.mean([df['time_s'].iloc[end] - df['time_s'].iloc[start] for start, end in live_indices])
    
    return [
        str(len(live_indices)), 
        str(total_num_samples), 
        str(total_time_duration), 
        str(average_test_duration), 
        live_indices
    ]


@callback(
    Output('test-selector-container', 'children'), 
    [Input('num-experiments-detected', 'children'), 
     Input('live-indices-store', 'data'), 
     Input('file-dropdown', 'value')]
)
def update_test_selector_container(num_experiments_detected, live_indices, selected_file): 
    """Dynamically generate test selectors based on the number of tests."""
    num_tests = int(num_experiments_detected)
    selectors = [] 
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    for i in range(0, num_tests + 1):        
        if i == 0: 
            test_str = "All Tests"
            num_samples = len(df)
            time_duration = df['time_s'].iloc[-1]
        else: 
            test_str = f"Test {i}"
            num_samples = live_indices[i - 1][1] - live_indices[i - 1][0]
            time_duration = df['time_s'].iloc[live_indices[i - 1][1]] - df['time_s'].iloc[live_indices[i - 1][0]]
        
        selectors.append(html.Div(
            children=[
                html.H3(test_str, className="text-lg font-bold"), 
                html.P(f"Num Samples: {num_samples}"), 
                html.P(f"Time Duration: {time_duration:.3f} s"), 
            ], 
            id={'type': 'test-selector', 'index': i}, 
            className="min-h-10 bg-gray-100 rounded-md p-2 hover:bg-gray-200 active:bg-gray-300 transition-colors duration-200 cursor-pointer"
        ))
    return selectors

@callback(
    Output('selected-test-index', 'data'), 
    [Input({'type': 'test-selector', 'index': ALL}, 'n_clicks')], 
    prevent_initial_call=True
)
def update_selected_test_index(*args): 
    """Update the selected test index when a test selector is clicked."""
    # print("Update selected test index")
    # print(args)
    # print("Context:")
    # print(ctx)
    # print(ctx.triggered) 
    # print(ctx.triggered_id)
    if not ctx.triggered: 
        return no_update
    clicked_id = ctx.triggered_id['index']
    return clicked_id
    


@callback(
    [Output('encoder-radians-plot', 'figure')], 
    [Input('file-dropdown', 'value'), 
     Input('selected-test-index', 'data'), 
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_encoder_radians(selected_file, selected_test_index, live_indices): 
    """Draws the encoder radians plot for the selected file and show options."""

    df = calibration_dfs[calibration_files.index(selected_file)]
    fig = go.Figure()
    
    if selected_test_index != 0: 
        fig.update_layout(title=f'Encoder Position - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else: 
        fig.update_layout(title='Encoder Position')
    
    
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians'], mode='lines', line_shape='hv', name='Raw'))
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians_smooth'], mode='lines', line_shape='hv', name='Smoothed'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Position (rad)')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    
    
    
    return [fig]

@callback(
    [Output('encoder-velocity-plot', 'figure')], 
    [Input('file-dropdown', 'value'),
     Input('selected-test-index', 'data'),
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_encoder_velocity(selected_file, selected_test_index, live_indices): 
    """Draws the encoder velocity plot for the selected file and show options."""
    
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    fig = go.Figure()
    
    if selected_test_index != 0:
        fig.update_layout(title=f'Encoder Velocity - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else:
        fig.update_layout(title='Encoder Velocity')
        
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians_dot'], mode='lines', line_shape='hv', name='Raw'))
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians_dot_smooth'], mode='lines', line_shape='hv', name='Smoothed'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Velocity (rad/s)')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    return [fig]
        
@callback(
    [Output('encoder-acceleration-plot', 'figure')],
    [Input('file-dropdown', 'value'),
     Input('selected-test-index', 'data'),
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_encoder_acceleration(selected_file, selected_test_index, live_indices):
    """Draws the encoder acceleration plot for the selected file and show options."""
    
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    fig = go.Figure()
    
    if selected_test_index != 0:
        fig.update_layout(title=f'Encoder Acceleration - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else:
        fig.update_layout(title='Encoder Acceleration')
        
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians_ddot'], mode='lines', line_shape='hv', name='Raw'))
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['EncoderRadians_ddot_smooth'], mode='lines', line_shape='hv', name='Smoothed'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Acceleration (rad/s²)')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    return [fig]

@callback(
    [Output('torque-plot', 'figure')],
    [Input('file-dropdown', 'value'),
     Input('selected-test-index', 'data'),
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_torque(selected_file, selected_test_index, live_indices):
    """Draws the torque plot for the selected file."""
    
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    fig = go.Figure()
    
    if selected_test_index != 0:
        fig.update_layout(title=f'Command Torque - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else:
        fig.update_layout(title='Command Torque')
        
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['CommandTorque'], mode='lines', line_shape='hv', name='Torque'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Torque (Nm)')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    return [fig]

@callback(
    [Output('grip-force-plot', 'figure')],
    [Input('file-dropdown', 'value'),
     Input('selected-test-index', 'data'),
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_grip_force(selected_file, selected_test_index, live_indices):
    """Draws the grip force plot for the selected file."""
    
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    fig = go.Figure()
    
    if selected_test_index != 0:
        fig.update_layout(title=f'Grip Force - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else:
        fig.update_layout(title='Grip Force')
        
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['GripForce'], mode='lines', line_shape='hv', name='Grip Force'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Force (N)')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    return [fig]

@callback(
    [Output('test-live-plot', 'figure')],
    [Input('file-dropdown', 'value'),
     Input('selected-test-index', 'data'),
     Input('live-indices-store', 'data')]
)
def draw_calibration_plot_test_live(selected_file, selected_test_index, live_indices):
    """Draws the test live status plot for the selected file."""
    
    df = calibration_dfs[calibration_files.index(selected_file)]
    
    fig = go.Figure()
    
    if selected_test_index != 0:
        fig.update_layout(title=f'Test Live Status - Test {selected_test_index}')
        
        # Filter the dataframe to only include the selected test
        df = df.iloc[live_indices[selected_test_index - 1][0]:live_indices[selected_test_index - 1][1]]
    else:
        fig.update_layout(title='Test Live Status')
        
    fig.add_trace(go.Scattergl(x=df['time_s'], y=df['TestIsLive'], mode='lines', line_shape='hv', name='Test Live Status'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='Status')

    # Add a legend
    fig.update_layout(legend=dict(x=0.8, y=0.95))
    return [fig]

if __name__ == "__main__": 
    app.run(debug=True) 