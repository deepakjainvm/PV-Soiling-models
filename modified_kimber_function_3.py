# This function calculates soiling ratio based on a modified version of the Kimber model
# It takes daily rain as input and returns a DataFrame with the calculated soiling ratio
import numpy as np
import pandas as pd

def modified_kimber_3(daily_rain, SR_threshold = 0.97, cleaning_threshold=0.508,
                            soiling_loss_rate=-0.0015, initial_soiling=0.97,
                            percent_loss_after_rain = -0.15, small_percent_gain_after_rain = 0.26,
                            big_percent_gain_after_rain = 1.29):
    """
    Parameters:
    - daily_rain: Dataframe containing daily rain data
    - SR_threshold: Threshold for determining soiling loss or gain based on soiling ratio
    - cleaning_threshold: Threshold for cumulative rain triggering cleaning
    - soiling_loss_rate: Rate of soiling loss in the absence of rain
    - initial_soiling: Initial soiling ratio
    - percent_loss_after_rain: Percent loss in soiling ratio after rain (negative)
    - small_percent_gain_after_rain: Small percent gain in soiling ratio after rain (positive)
    - big_percent_gain_after_rain: Big percent gain in soiling ratio after rain (positive)
    
    Returns:
    - DataFrame with the calculated soiling ratio
    """
    
    # Assumptions SR of less than 97: 1.29; SR >97: 0.26; Rainfall< 0.508: -0.15 
    value_1 = percent_loss_after_rain / 100
    value_2 = small_percent_gain_after_rain / 100
    value_3 = big_percent_gain_after_rain / 100
   
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
            # If rain, update soiling ratio based on cumulative rain and soiling thresholds
            df['SR'][idx] = np.nan
            rain_cum += df['Daily Rain'][idx]
            if rain_cum <= cleaning_threshold:
                temp_soiling_loss_rate = SR * value_1
            else:
                if SR > SR_threshold:
                    temp_soiling_loss_rate = SR * value_2
                else:
                    temp_soiling_loss_rate = SR * value_3

    # Replace 0 with NaN in the 'SR' column
    df['SR'].replace(0, np.nan, inplace=True)
    df['SR'].ffill(inplace=True)
    return df[['SR']]



