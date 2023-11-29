# This program modifies the kimber model such that the SR never goes back to one after the threshold rain
# It has slope and intercept for gain after rain. Note: A negative slope is used, as a higher previous day soiling
# ratio results in a lower gain after rain, and vice versa.
import numpy as np
import pandas as pd


def modified_kimber(daily_rain, cleaning_threshold=0.508,
                            soiling_loss_rate=-0.0015, initial_soiling=0.97,
                            slope = -0.002, intercept = 0.004):
    
    """
   Modified Kimber Model for Soiling Ratio Calculation

   Parameters:
   - daily_rain: Pandas Series or DataFrame containing daily rainfall data.
   - cleaning_threshold: Threshold for rain accumulation triggering cleaning.
   - soiling_loss_rate: Rate of soiling loss in the absence of rain
   - initial_soiling: Initial soiling ratio.
   - slope: Slope parameter for gain after rain.
   - intercept: Intercept parameter for gain after rain.

   Returns:
   - Pandas DataFrame with 'SR' column representing the calculated soiling ratio.
   """
   
    # Initializations
    temp_soiling_loss_rate = soiling_loss_rate
    rain_cum = 0
    df = pd.DataFrame(daily_rain.copy())
    df.columns = ['Daily Rain']
    df['SR'] = np.nan
    df.iloc[0, 1] = initial_soiling
    SR = df['SR'][0]
   
    # Calculations
    for idx in df.index[1:]:
        if df['Daily Rain'][idx] == 0:
            # If no rain, update soiling ratio based on the soiling loss rate
            df.loc[idx, 'SR'] = SR + temp_soiling_loss_rate  
            SR = df['SR'][idx]
            rain_cum = 0
            temp_soiling_loss_rate = soiling_loss_rate
        else:
            # If rain, set soiling ratio to 0 and update soiling loss rate based on cumulative rain and slope
            df['SR'][idx] = 0
            rain_cum += df['Daily Rain'][idx]
            if rain_cum <= cleaning_threshold:
                temp_soiling_loss_rate = 0
            else:
                temp_soiling_loss_rate = SR * (slope * SR + intercept)
                
    # Replace 0 with NaN in the 'SR' column
    df['SR'].replace(0, np.nan, inplace=True)
    df['SR'].ffill(inplace=True)
    return df[['SR']]





