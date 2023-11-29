# Function to calculate modified hsu soiling which modifies hsu soiling to include other parameters
# It takes daily rain, PM2.5 and PM10 as input and returns a DataFrame with the calculated soiling ratio

import datetime
import numpy as np
import pandas as pd
from scipy.special import erf
from pvlib.tools import cosd

def modified_hsu(rainfall, cleaning_threshold, surface_tilt, pm2_5, pm10,
        depo_veloc=None, rain_accum_period=pd.Timedelta('24h'),
        SR_threshold = 0.97,initial_soiling=0.97,
        small_percent_gain_after_rain = 0.26,big_percent_gain_after_rain = 1.29):
    
    """
   Modified HSU Model for Soiling Ratio Calculation

   Parameters:
   - rainfall: Pandas Series or DataFrame containing rainfall data.
   - cleaning_threshold: Threshold for rain accumulation triggering cleaning.
   - surface_tilt: Tilt of the surface.
   - pm2_5: Mass concentration of PM2.5 particles.
   - pm10: Mass concentration of PM10 particles.
   - depo_veloc: Deposition velocity for particles (default values provided).
   - rain_accum_period: Time period for accumulating rainfall.
   - SR_threshold: Soiling ratio threshold.
   - initial_soiling: Initial soiling ratio.
   - small_percent_gain_after_rain: Small percent gain in soiling ratio after rain (positive)
   - big_percent_gain_after_rain: Big percent gain in soiling ratio after rain (positive)

   Returns:
   - Pandas DataFrame with 'SR' column representing the calculated soiling ratio.
   """
   
    df = pd.DataFrame(rainfall.copy())
    df.columns = ['Daily Rain']
    if depo_veloc is None:
        depo_veloc = {'2_5': 0.0009, '10': 0.004}
    # accumulate rainfall into periods for comparison with threshold
    accum_rain = rainfall.rolling(rain_accum_period, closed='right').sum()
    # cleaning is True for intervals with rainfall greater than threshold
    cleaning_times = accum_rain.index[accum_rain >= cleaning_threshold]

    # determine the time intervals in seconds (dt_sec)
    dt = rainfall.index
    # subtract shifted values from original and convert to seconds
    dt_diff = (dt[1:] - dt[:-1]).total_seconds()
    # ensure same number of elements in the array, assuming that the interval
    # prior to the first value is equal in length to the first interval
    dt_sec = np.append(dt_diff[0], dt_diff).astype('float64')

    horiz_mass_rate = (
        pm2_5 * depo_veloc['2_5'] + np.maximum(pm10 - pm2_5, 0.)
        * depo_veloc['10']) * dt_sec
    tilted_mass_rate = horiz_mass_rate * cosd(surface_tilt)  # assuming no rain

    # tms -> tilt_mass_rate
    tms_cumsum = np.cumsum(tilted_mass_rate * np.ones(rainfall.shape))

    mass_no_cleaning = pd.Series(index=rainfall.index, data=tms_cumsum)
    # specify dtype so pandas doesn't assume object
    mass_removed = pd.Series(index=rainfall.index, dtype='float64')
    mass_removed[0] = 0.
    mass_removed[cleaning_times] = mass_no_cleaning[cleaning_times]
    accum_mass = mass_no_cleaning - mass_removed.ffill()

    soiling_rate_accumulation = -0.3437 * erf(0.17 * accum_mass**0.8473)
    soiling_rate = soiling_rate_accumulation.diff()
    soiling_rate = soiling_rate.shift(-1)
    soiling_rate = soiling_rate.ffill()
    soiling_rate = soiling_rate.rename('Soiling_Rate')
    
    # combine soiling ratio to df
    df = df.join(soiling_rate)
    
    #%%
    # Assumptions SR of less than 97: 1.29; SR >97: 0.26; Rainfall< 0.508: -0.15 
    value_1 = small_percent_gain_after_rain / 100
    value_2 = big_percent_gain_after_rain / 100
    temp_soiling_loss_rate = soiling_rate_accumulation[0]
    rain_cum = 0
    df['SR'] = 0  
    df.iloc[0, 2] = initial_soiling
    SR = df['SR'][0]
    
    #%%
    # calculations
    for idx in df.index[0:]:
        if df['Daily Rain'][idx] == 0:
            # If no rain, update soiling ratio based on the soiling loss rate
            df.loc[idx, 'SR'] = SR + temp_soiling_loss_rate
            SR = df['SR'][idx]
            rain_cum = 0
            temp_soiling_loss_rate = soiling_rate[idx]
        else:
            # If rain, update soiling ratio based on cumulative rain and soiling thresholds
            df['SR'][idx] = 0
            rain_cum += df['Daily Rain'][idx]
            if rain_cum <= cleaning_threshold:
                temp_soiling_loss_rate = 0
            else:
                if (SR > SR_threshold):
                    temp_soiling_loss_rate = SR * value_1
                else:
                    temp_soiling_loss_rate = SR * value_2
      
    #%% Replace 0 with NaN in the 'SR' column
    df['SR'].replace(0, np.nan, inplace=True)
    df['SR'].ffill(inplace=True)
    return df[['SR']]



    
    
    
    
    
    
    
    
    
    
    
    
    