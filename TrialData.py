import pandas as pd
import numpy as np 
from typing import Union
class CalibrationTrial(pd.DataFrame): 
    _metadata = ['outlier']
    
    def __init__(self, *args, **kwargs): 
        self.outlier = False
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self): 
        return CalibrationTrial
    
    def mark_outlier(self): 
        self.outlier = True
        
    def unmark_outlier(self): 
        self.outlier = False
        
    def is_outlier(self): 
        return self.outlier
    
    def is_valid(self): 
        return not self.outlier
        
    def calculate_metrics(self, max_time : float = 0.1): 
        """
        Calculate the m/b/k estimates for this trial. 

        Return a dictionary with the following keys: 
        - 'inertia' : float
        - 'damping' : float
        - 'stiffness' : float
        - 'residual_error' : float
        - 'grip_force_threshold' : float
        """    
        
        results = {} 
        
        thetas = self['EncoderRadians_smooth'][self['elapsed_time'] <= max_time]
        thetas_dot = self['EncoderRadians_dot_smooth'][self['elapsed_time'] <= max_time]
        thetas_ddot = self['EncoderRadians_ddot_smooth'][self['elapsed_time'] <= max_time]
        torques = self['CommandTorque'][self['elapsed_time'] <= max_time]
        
        # Now form matrices
        A = np.column_stack((thetas_ddot, thetas_dot, thetas))
        b = torques 
        
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        results['inertia'] = x[0]
        results['damping'] = x[1]
        results['stiffness'] = x[2]
        results['residual_error'] = residuals[0]
        results['grip_force_threshold'] = self['GripThreshold'].iloc[0]
        
        return results 
    
    
class MainTrial(pd.DataFrame): 
    _metadata = ['showLoads', 'outlier', 'corrected_outlier', '_is_catch_trial', 'previous_trial_load']
    
    def __init__(self, *args, **kwargs): 
        self.showLoads = False
        self.outlier = False
        self.corrected_outlier = False
        self._is_catch_trial = False
        self.previous_trial_load = None
        super().__init__(*args, **kwargs)
        
    @property
    def _constructor(self): 
        return MainTrial
    
    def set_show_loads(self, show_loads : bool): 
        self.showLoads = show_loads
        
    def is_load_visible(self): 
        return self.showLoads
    
    def get_load(self): 
        # Return max applied torque
        return abs(self['CommandTorque']).max()

    def set_previous_trial_load(self, load: Union[float, None]): 
        self.previous_trial_load = load
        self._is_catch_trial = self.previous_trial_load != self.get_load() and self.previous_trial_load is not None
        
    def get_previous_trial_load(self): 
        return self.previous_trial_load
    
    def is_catch_trial(self): 
        return self._is_catch_trial
    
    def get_previous_trial_load(self): 
        return self.previous_trial_load
        
    def mark_outlier(self): 
        self.outlier = True
        
    def unmark_outlier(self): 
        self.outlier = False
        
    def is_outlier(self): 
        return self.outlier
    
    def is_valid(self): 
        return not self.outlier
    
    def mark_corrected_outlier(self): 
        self.corrected_outlier = True
        
    def unmark_corrected_outlier(self): 
        self.corrected_outlier = False
        
    def is_corrected_outlier(self): 
        return self.corrected_outlier
    
    def correct_zero_position(self): 
        """ 
        Corrects the zero position of the trial to the first encoder position. 
        """
        self['EncoderRadians_smooth'] = self['EncoderRadians_smooth'] - self['EncoderRadians_smooth'].iloc[0]

    def correct_start_index(self, start_index: int) -> None:
        """
        Corrects the trial by adjusting the start index and recalculating elapsed time.
        This modifies the DataFrame in-place.
        
        Args:
            start_index: The new starting index to use
        """
        # Get the indices to keep
        indices_to_keep = self.index[start_index:]
        
        # Drop the rows we don't want to keep
        indices_to_drop = self.index.difference(indices_to_keep)
        self.drop(indices_to_drop, inplace=True)
        
        # Reset the index
        self.reset_index(drop=True, inplace=True)
        
        # Recalculate elapsed time
        self['elapsed_time'] = self['time_s'] - self['time_s'].min()
        
        # Mark as corrected
        self.mark_corrected_outlier()