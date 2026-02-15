# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:06:32 2020

@author: kerschensteinerlab
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tkinter import filedialog
from tkinter import *
import sys
from PyQt5.QtWidgets import (QLabel, QRadioButton, QVBoxLayout, QApplication, QWidget)
import os
import matplotlib.image as mpimg 
import math
import cv2


def h5_to_df(h5_file: str, frame_rate: float = 30) -> pd.DataFrame:
    """Convert DeepLabCut HDF5 tracking file to pandas DataFrame.
    
    Extracts pose tracking data for mouse body parts (nose, ears, tail base) and cricket,
    including x/y coordinates and likelihood scores for each tracked point.
    
    Parameters
    ----------
    h5_file : str
        Path to DeepLabCut .h5 output file
    frame_rate : float, optional
        Video frame rate in fps (default: 30)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - frame_number, time
        - Body part positions: {bodypart}_{x,y,likelihood}
        - Computed mid-point between ears: mid_{x,y}
    
    Notes
    -----
    Requires DeepLabCut tracking with body parts: 
    'cricket', 'nose', 'l_ear', 'r_ear', 'tail_base'
    """
    
    pos = pd.read_hdf(h5_file) 
    network_name = pos.columns[0][0]
    
    
    cricket_x = np.squeeze(np.array(pos[network_name, 'cricket','x']))
    cricket_y = np.squeeze(np.array(pos[network_name, 'cricket','y']))
    cricket_likelihood = np.squeeze(np.array(pos[network_name, 'cricket','likelihood']))
    nose_x = np.squeeze(np.array(pos[network_name, 'nose','x']))
    nose_y = np.squeeze(np.array(pos[network_name, 'nose','y']))
    nose_likelihood = np.squeeze(np.array(pos[network_name, 'nose','likelihood']))
    leftear_x = np.squeeze(np.array(pos[network_name, 'l_ear','x']))
    leftear_y = np.squeeze(np.array(pos[network_name, 'l_ear','y']))
    leftear_likelihood = np.squeeze(np.array(pos[network_name, 'l_ear','likelihood']))
    rightear_x = np.squeeze(np.array(pos[network_name, 'r_ear','x']))
    rightear_y = np.squeeze(np.array(pos[network_name, 'r_ear','y']))
    rightear_likelihood = np.squeeze(np.array(pos[network_name, 'r_ear','likelihood']))
    tailbase_x = np.squeeze(np.array(pos[network_name, 'tail_base','x']))
    tailbase_y = np.squeeze(np.array(pos[network_name, 'tail_base','y']))
    tailbase_likelihood = np.squeeze(np.array(pos[network_name, 'tail_base','likelihood']))
    mid_x = np.squeeze(np.array(np.mean([leftear_x,rightear_x],axis=0)))
    mid_y = np.squeeze(np.array(np.mean([leftear_y,rightear_y],axis=0)))
    
    n_frames = len(cricket_x)
    frames = np.squeeze(np.linspace(0,n_frames-1, n_frames))
    total_time = n_frames/frame_rate
    time = np.linspace(0,total_time,n_frames)
       
    df = pd.DataFrame({'frame_number': frames,
                       'time': time, 
                       'leftear_x': leftear_x,
                       'leftear_y': leftear_y,
                       'leftear_likelihood':leftear_likelihood,
                       'rightear_x': rightear_x,
                       'rightear_y': rightear_y,
                       'rightear_likelihood':rightear_likelihood,
                       'mid_x': mid_x,
                       'mid_y': mid_y,
                       'nose_x': nose_x,
                       'nose_y': nose_y,
                       'nose_likelihood': nose_likelihood,
                       'tailbase_x': tailbase_x,
                       'tailbase_y': tailbase_y,
                       'tailbase_likelihood': tailbase_likelihood,
                       'cricket_x': cricket_x,
                       'cricket_y': cricket_y,
                       'cricket_likelihood': cricket_likelihood
                       })
            
    return df


def calculate_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate midpoint between left and right ears (vectorized).
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with 'leftear_x', 'leftear_y', 'rightear_x', 'rightear_y'
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'mid_x' and 'mid_y' columns
        
    Notes
    -----
    This vectorized implementation is ~100-1000x faster than the original row-by-row loop.
    """
    df['mid_x'] = (df['leftear_x'] + df['rightear_x']) / 2
    df['mid_y'] = (df['leftear_y'] + df['rightear_y']) / 2
    
    return df


def calculate_head(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate head centroid position from nose and ears.
    
    Computes the center of mass of the head as the average of nose, 
    left ear, and right ear positions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with nose and ear positions
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'head_x' and 'head_y' columns
    """
    df['head_x'] = np.mean(np.array([df['leftear_x'], df['rightear_x'], df['nose_x']]), axis=0)
    df['head_y'] = np.mean(np.array([df['leftear_y'], df['rightear_y'], df['nose_y']]), axis=0)
    print('head centroid position columns [head_x and head_y] added')
    
    return df


def interpolate_unlikely_label_positions(df, likelihood_cutoff = 0.9, cricket=True, nose=True, tailbase=True):
    
    if cricket:
        df.loc[df['cricket_likelihood'] < likelihood_cutoff, ['cricket_x', 'cricket_y']] = np.nan
        df[['cricket_x', 'cricket_y']] = df[['cricket_x', 'cricket_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely cricket label positions linearly interpolated')
    
    if nose: 
        df.loc[df['nose_likelihood'] < likelihood_cutoff, ['nose_x', 'nose_y']] = np.nan
        df[['nose_x', 'nose_y']] = df[['nose_x', 'nose_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely nose label positions linearly interpolated')
    
    if tailbase: 
        df.loc[df['tailbase_likelihood'] < likelihood_cutoff, ['tailbase_x', 'tailbase_y']] = np.nan
        df[['tailbase_x', 'tailbase_y']] = df[['tailbase_x', 'tailbase_y']].interpolate(method='linear', limit_direction='both')
        print('unlikely tailbase label positions linearly interpolated')
    
    return df


def smooth_labels(df, smooth_frames=15, smooth_order=3):
    
    df[['leftear_x', 'leftear_y','rightear_x', 'rightear_y','nose_x', 'nose_y','tailbase_x', 'tailbase_y','cricket_x', 'cricket_y']] = savgol_filter(df[['leftear_x', 'leftear_y','rightear_x', 'rightear_y','nose_x', 'nose_y','tailbase_x', 'tailbase_y','cricket_x', 'cricket_y']], smooth_frames, smooth_order,axis=0)
    
    print('label positions smoothed')
    
    return df
   

def lineardistance(x1: float, x2: np.ndarray, y1: float, y2: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance between point (x1, y1) and array of points (x2, y2).
    
    Parameters
    ----------
    x1 : float
        X-coordinate of first point
    x2 : np.ndarray
        X-coordinates of second points
    y1 : float
        Y-coordinate of first point
    y2 : np.ndarray
        Y-coordinates of second points
        
    Returns
    -------
    np.ndarray
        Euclidean distances
    """
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def get_distance_to_borders(df, borders):
    
    cricket_to_borders = np.zeros((len(df),1))
    mouse_to_borders = np.zeros((len(df),1))
    
    for i in range(len(df)):
        cricket_to_borders[i] = np.min(lineardistance(df['cadj_x'][i], borders[:,0], df['cadj_y'][i], borders[:,1]))
        mouse_to_borders[i] = np.min(lineardistance(df['madj_x'][i], borders[:,0], df['madj_y'][i], borders[:,1]))
    
    df['cricket_to_borders'] = cricket_to_borders
    df['mouse_to_borders'] = mouse_to_borders
    print('mouse_to_borders and cricket_to_borders columns added')
        
    return df


def get_distance_path_to_borders(df, borders, n_samples=100):
    
    path_to_borders = np.zeros((len(df),1))
    
    for i in range(len(df)):
        path_x = np.linspace(df['madj_x'][i], df['cadj_x'][i], n_samples)
        path_y = np.linspace(df['madj_y'][i], df['cadj_y'][i], n_samples)
        path_distance = np.zeros((n_samples, 1))

        for j in range(n_samples):
            path_distance[j] = np.min(lineardistance(path_x[j], borders[:,0], path_y[j], borders[:,1]))
        
        path_to_borders[i] = np.mean(path_distance)
    
    df['path_to_borders'] = path_to_borders
    print('path_to_borders column added')
            
    return df


def get_azimuth_head(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate head azimuth angle relative to cricket.
    
    Computes the angle between the head direction (mid-ears to nose) and 
    the cricket direction (mid-ears to cricket). Uses law of cosines.
    Sign is negative if cricket is on the left, positive if on the right.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with mouse and cricket positions
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'azimuth_head' column (degrees)
    """
    
    a = lineardistance(df['cricket_x'], df['mid_x'], df['cricket_y'], df['mid_y'])
    b = lineardistance(df['cricket_x'], df['nose_x'], df['cricket_y'], df['nose_y'])
    c = lineardistance(df['nose_x'], df['mid_x'], df['nose_y'], df['mid_y'])    

    B = np.degrees(np.arccos(((c**2)+(a**2)-(b**2))/(2*c*a)))

    azimuth = B
    leftear_distance = lineardistance(df['cricket_x'], df['leftear_x'], df['cricket_y'], df['leftear_y'])
    rightear_distance = lineardistance(df['cricket_x'], df['rightear_x'], df['cricket_y'], df['rightear_y'])
    
    azimuth[leftear_distance < rightear_distance] = -azimuth[leftear_distance < rightear_distance]
    
    df['azimuth_head'] = azimuth    
    print('azimuth_head column added to dataframe')
    
    return df


def get_azimuth_body(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate body azimuth angle relative to cricket.
    
    Computes the angle between the body axis (tail base to mid-ears) and 
    the cricket direction (mid-ears to cricket). Uses law of cosines.
    Sign is negative if cricket is on the left, positive if on the right.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with mouse and cricket positions
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with added 'azimuth_body' column (degrees)
    """
    
    a = lineardistance(df['cricket_x'], df['tailbase_x'], df['cricket_y'], df['tailbase_y'])
    b = lineardistance(df['cricket_x'], df['mid_x'], df['cricket_y'], df['mid_y'])
    c = lineardistance(df['mid_x'], df['tailbase_x'], df['mid_y'], df['tailbase_y'])
    
    B = np.degrees(np.arccos(((c**2)+(a**2)-(b**2))/(2*c*a)))

    azimuth = B
    leftear_distance = lineardistance(df['cricket_x'], df['leftear_x'], df['cricket_y'], df['leftear_y'])
    rightear_distance = lineardistance(df['cricket_x'], df['rightear_x'], df['cricket_y'], df['rightear_y'])
    
    azimuth[leftear_distance < rightear_distance] = -azimuth[leftear_distance < rightear_distance]
    
    df['azimuth_body'] = azimuth    
    print('azimuth_body column added to dataframe')
    
    return df


def get_azimuth_head_body(df):
    
    a = lineardistance(df['nose_x'] + df['tailbase_x'] - df['mid_x'], df['tailbase_x'], df['nose_y'] + df['tailbase_y'] - df['mid_y'],df['tailbase_y'])
    b = lineardistance(df['nose_x'] + df['tailbase_x'] - df['mid_x'], df['mid_x'], df['nose_y'] + df['tailbase_y'] - df['mid_y'], df['mid_y'])
    c = lineardistance(df['mid_x'],df['tailbase_x'],df['mid_y'],df['tailbase_y'])
    
    B = np.degrees(np.arccos(((c**2)+(a**2)-(b**2))/(2*c*a)))

    azimuth = B
    leftear_distance = lineardistance(df['tailbase_x'],df['leftear_x'],df['tailbase_y'],df['leftear_y'])
    rightear_distance = lineardistance(df['tailbase_x'],df['rightear_x'],df['tailbase_y'],df['leftear_y'])
    
    azimuth[leftear_distance < rightear_distance] = -azimuth[leftear_distance < rightear_distance]
    
    df['azimuth_head_body'] = azimuth    
    print('azimuth_head_body column added to dataframe')
    
    return df


def get_azimuth_head_arena(df):
    
    heading_x = np.array(df['nose_x'] - df['mid_x'])
    heading_y = np.array(df['nose_y'] - df['mid_y'])
    
    heading_angle = np.empty(len(heading_x))
    heading_angle[:] = np.nan
    
    for i in range(len(heading_x)-1):
        heading_current = np.array([heading_x[i],heading_y[i]]) / np.linalg.norm(np.array([heading_x[i],heading_y[i]]))
        heading_future = np.array([heading_x[i+1],heading_y[i+1]]) / np.linalg.norm(np.array([heading_x[i+1],heading_y[i+1]]))
        
        heading_angle[i] = np.degrees(np.arccos(np.clip(np.dot(heading_current, heading_future), -1.0, 1.0)))
        
    
    df['azimuth_head_arena'] = heading_angle    
    print('azimuth_head_arena column added to dataframe')
    
    return df


def get_azimuth_body_arena(df):
    
    bearing_x = np.array(df['mid_x'] - df['tailbase_x'])
    bearing_y = np.array(df['mid_y'] - df['tailbase_y'])
    
    bearing_angle = np.empty(len(bearing_x))
    bearing_angle[:] = np.nan
    
    for i in range(len(bearing_x)-1):
        bearing_current = np.array([bearing_x[i],bearing_y[i]]) / np.linalg.norm(np.array([bearing_x[i],bearing_y[i]]))
        bearing_future = np.array([bearing_x[i+1],bearing_y[i+1]]) / np.linalg.norm(np.array([bearing_x[i+1],bearing_y[i+1]]))
        
        bearing_angle[i] = np.degrees(np.arccos(np.clip(np.dot(bearing_current, bearing_future), -1.0, 1.0)))
        
    
    df['azimuth_body_arena'] = bearing_angle    
    print('azimuth_body_arena column added to dataframe')
    
    return df


def get_distance_to_cricket(df, cm_per_pixel=0.23):

    df['cricket_distance'] = lineardistance(df['cricket_x'],df['mid_x'],df['cricket_y'],df['mid_y'])*cm_per_pixel
    print('cricket_distance column added to dataframe')
    
    return df


def get_mouse_speed(df, frame_rate=30, cm_per_pixel=0.23, smooth_frames=15, smooth_order=3):

    delta_mid_x = np.diff(df['tailbase_x'],axis=0)# switched to tailbase from mid_x
    delta_mid_y = np.diff(df['tailbase_y'],axis=0)# switched to tailbase from mid_y
     
    position_change = np.sqrt(delta_mid_x**2 + delta_mid_y**2)*cm_per_pixel
     
    speed = np.array([0])
    speed = np.append(speed,position_change*frame_rate*.5)
    
    speed = savgol_filter(speed,smooth_frames,smooth_order,axis=0)
    
    speed[speed < 0] = 0
    
    df['mouse_speed'] = speed    
    print('mouse_speed column added to dataframe')
    
    return df


def get_mouse_acceleration(df, smooth_frames=15, smooth_order=3):
    
    acceleration = np.array([0])
    acceleration = np.append(acceleration, np.array(df['mouse_speed'].diff().dropna()))
    acceleration = savgol_filter(acceleration,smooth_frames,smooth_order,axis=0)
    acceleration[0:1] = 0
    
    df['mouse_acceleration'] = acceleration
    
    return df
       

def get_cricket_speed(df, frame_rate=30, cm_per_pixel=0.23, smooth_frames=15, smooth_order=3):
    
    delta_cricket_x = np.diff(df['cricket_x'],axis=0)
    delta_cricket_y = np.diff(df['cricket_y'],axis=0)
     
    position_change = np.sqrt(delta_cricket_x**2 + delta_cricket_y**2)*cm_per_pixel
     
    speed = np.array([0])
    speed = np.append(speed,position_change*frame_rate*.5)
    
    
    speed[speed < 0] = 0
    
    speed = savgol_filter(speed,smooth_frames,smooth_order,axis=0)
    
    df['cricket_speed'] = speed    
    print('cricket_speed column added to dataframe')
    
    return df


def get_contacts(df, contact_distance = 4):
    
    time_in_contact =np.zeros_like(df['time'])
    time_in_contact[df['cricket_distance'] < contact_distance] = 1

    df['contact'] = time_in_contact    
    print('contact column added to dataframe')
    
    return df


def smooth_contacts(df: pd.DataFrame, window_size: int = 8) -> pd.DataFrame:
    """Smooth contact detection using a moving average filter.
    
    Applies a uniform moving average to reduce noise in contact detection.
    Values >= 1/window_size are set to 1 (contact), others to 0.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with 'contact' column
    window_size : int, optional
        Window size for moving average (default: 8)
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with smoothed 'contact' column
    """
    
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(np.squeeze(df['contact']), weights, 'same')
    yMA[yMA >= 1/window_size] = 1
    yMA[yMA < 1/window_size] = 0
    time_in_contact_smooth = yMA
    
    df['contact'] = time_in_contact_smooth    
    print('contact column smoothed')
    
    return df


def get_approaches(df, speed_threshold=10, diff_frames=4, diff_speed=-20, frame_rate=30, body_azimuth=60): #diff_frames=8, diff_speed=-1 no frame_rate
    
    dist_diff = df['cricket_distance'].diff().rolling(diff_frames).mean()
    
    #df['approach'] = (df['mouse_speed'].rolling(diff_frames).mean() > speed_threshold) & (abs(df['azimuth'].rolling(diff_frames).mean()) < 90) & (dist_diff < diff_speed) #misplacecd brackets mean the abs was calculated after azimuth mean causing +/- to cancel
    
    df['approach'] = (df['mouse_speed'].rolling(diff_frames).mean() > speed_threshold) & (abs(df['azimuth_body']).rolling(diff_frames).mean() < body_azimuth) & (dist_diff < diff_speed/frame_rate)
    
    df['approach'] = df['approach'].astype(int)
    
    return df


def test_dist_diff(df,diff_frames=8):
    
    dist_diff = df['cricket_distance'].diff().rolling(diff_frames).mean()
    
    plt.plot(dist_diff)
    
    return


def smooth_approaches(df: pd.DataFrame, window_size: int = 8) -> pd.DataFrame:
    """Smooth approach detection using a moving average filter.
    
    Applies a uniform moving average to reduce noise in approach detection.
    Values >= 1/window_size are set to 1 (approach), others to 0.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tracking dataframe with 'approach' column
    window_size : int, optional
        Window size for moving average (default: 8)
        
    Returns
    -------
    pd.DataFrame
        Input dataframe with smoothed 'approach' column
    """
    
    weights = np.repeat(1.0, window_size) / window_size
    yMA = np.convolve(np.squeeze(df['approach']), weights, 'same')
    yMA[yMA >= 1/window_size] = 1
    yMA[yMA < 1/window_size] = 0
    time_approaching_smooth = yMA
    
    df['approach'] = time_approaching_smooth    
    print('approach column smoothed')
    
    return df


def adjust_approaches(df, window_size = 8):
    
    clip_frames = window_size-1
    
    starts = get_approach_start_indices(df, to_capture_time=True)
    
    for approach in starts:
        df['approach'][approach:approach+clip_frames] = 0   
    
    return df


def get_interapproach_intervals(df, to_capture_time=True):
    
    starts = get_approach_start_indices(df, to_capture_time)
    ends = get_approach_end_indices(df, to_capture_time)
    
    if starts.size < 2:
        return 0
    else:
        starts = starts[1:]
        ends = ends[:-1]
    
    return starts-ends

def get_approach_intervals(df, to_capture_time=True, fps=30):
    
    approach_starts = get_approach_start_indices(df, to_capture_time)
    approach_ends = get_approach_end_indices(df, to_capture_time)
    contact_ends = get_contact_end_indices(df, to_capture_time)
    
    if approach_starts.size <=1:
        summary_approach_intervals = np.array(0)
    else:
        n_approaches = len(approach_starts)
        approach_intervals = np.zeros((n_approaches,1))
        for i in range(n_approaches):
            if i > 0:
                last_approach_end = approach_ends[i-1]
            else:
                last_approach_end = 0
                
            all_previous_contact_ends = np.argwhere(contact_ends < approach_starts[i])
            if all_previous_contact_ends.shape[0] > 0:
                last_contact_end = all_previous_contact_ends[-1][0]
            else:
                last_contact_end = 0
            
            approach_intervals[i] = np.min([approach_starts[i]-last_approach_end, approach_starts[i]-last_contact_end]) * 1/fps
            
        summary_approach_intervals = np.array([np.median(approach_intervals), np.mean(approach_intervals)])
                                
    return summary_approach_intervals



def get_median_interapproach_interval(df, to_capture_time=True, fps=30):
    
    return (1/fps)*np.median(get_interapproach_intervals(df, to_capture_time=True))


def get_mean_interapproach_interval(df, to_capture_time=True, fps=30):
    
    return (1/fps)*np.mean(get_interapproach_intervals(df, to_capture_time=True))


def get_approach_path_to_border(df, to_capture_time=True):
    
    approach_starts = get_approach_start_indices(df, to_capture_time)
    median_approach_to_border = np.median(df['path_to_borders'][approach_starts])
    mean_approach_to_border = np.mean(df['path_to_borders'][approach_starts])
    
    return median_approach_to_border, mean_approach_to_border


def get_approach_contact_latency(df, to_capture_time=True, fps=30):
    
    df['approach'][(df['contact']==1) & (df['approach']==1)] = 0
    
    approach_ends = get_approach_end_indices(df, to_capture_time=True)
    contact_start = get_contact_start_indices(df, to_capture_time=True)
    
    approach_contact_latency = []
    
    if approach_ends.size < 2:
        
        try:
            conts = contact_start[contact_start >= approach_ends]
            return min(conts)-approach_ends
        except:
            return 1
    
    for num in approach_ends:
        
        conts = contact_start[contact_start > num]
        
        try: 
            approach_contact_latency.append(min(conts)-num)
        except:
            continue
            
    
    return approach_contact_latency


def remove_short_approaches(df, min_approach_frames = 8, min_approach_length= 10, min_approach_distance = 10):
    
# =============================================================================
#     approach_frames = get_approach_end_indices(df) - get_approach_start_indices(df)
#     short_approaches = np.squeeze(np.argwhere(approach_frames <= min_approach_frames))
#     
#     if short_approaches.size > 1:
#         count = 0
#         for approach in short_approaches:
#             df = remove_approach(df, approach - count)
#             count = count+1
#     else:
#         df = remove_approach(df, short_approaches)
# =============================================================================
        
    starts = get_approach_start_indices(df)
    ends = get_approach_end_indices(df)
    if starts.size > 1:
        for s, e in zip(starts, ends):
            if e-s < min_approach_frames:
                df['approach'][s:e] = 0
       
    starts = get_approach_start_indices(df)
    ends = get_approach_end_indices(df)
    if starts.size > 1:
        for s, e in zip(starts, ends):
            if get_approach_length(df, s, e) <= min_approach_length:
                df['approach'][s:e] = 0
            
    starts = get_approach_start_indices(df)
    ends = get_approach_end_indices(df)
    if starts.size > 1:
        for s, e in zip(starts, ends):
            if get_min_approach_distance(df, s, e) >= min_approach_distance:
                df['approach'][s:e] = 0   
    
    return df


def remove_short_contacts(df, min_contact_frames = 3):
    
    starts = get_contact_start_indices(df)
    ends = get_contact_end_indices(df)
    if starts.size > 1:
        for s, e in zip(starts, ends):
            if e-s < min_contact_frames:
                df['contact'][s:e] = 0
    
    return df


def remove_approach_when_contact(df):
    
    df['approach'][(df['contact']==1) & (df['approach']==1)] = 0    
    
    return df


def get_approach_length(df, start_index, end_index):
    #print(get_approach_start_indices(df)[approach_index])
    return df['cricket_distance'][start_index:end_index].diff().abs().sum()


def get_min_approach_distance(df, start_index, end_index):
    
    return df['cricket_distance'][start_index:end_index].min()


def get_max_approach_distance(df, start_index, end_index):
    
    return df['cricket_distance'][start_index:end_index].max()


def remove_contact(df, contact_index, to_capture_time = True):
    
    starts = get_contact_start_indices(df, to_capture_time)
    ends = get_contact_end_indices(df, to_capture_time)
    df['contact'][starts[contact_index]:ends[contact_index]] = 0 
    
    return df


def remove_approach(df, approach_index, to_capture_time = True):
    
    starts = get_approach_start_indices(df, to_capture_time)
    ends = get_approach_end_indices(df, to_capture_time)
    df['approach'][starts[approach_index]:ends[approach_index]] = 0
    
    return df


def adjust_contacts(df, window_size = 8):
    
    clip_frames = window_size-1
    
    starts = get_contact_start_indices(df, to_capture_time=True)
    
    for contact in starts:
        df['contact'][contact:contact+clip_frames] = 0
    
    
    
    return df


def get_intercontact_intervals(df, to_capture_time=True):
    
    starts = get_contact_start_indices(df, to_capture_time)
    ends = get_contact_end_indices(df, to_capture_time)
    
    if starts.size < 2:
        return 0
    else:
        starts = starts[1:]
        ends = ends[:-1]
    
    return starts-ends


def get_median_intercontact_interval(df, to_capture_time=True, fps = 30):
    
    return (1/fps)*np.median(get_intercontact_intervals(df, to_capture_time=True))


def get_mean_intercontact_interval(df, to_capture_time=True, fps = 30):
    
    return (1/fps)*np.mean(get_intercontact_intervals(df, to_capture_time=True))


def add_pre_contact(df, pre_frames = 15):
    
    df['pre_contact'] = 0
    
    contact_starts = get_contact_start_indices(df)
    pre_contact = contact_starts - pre_frames
    
    if contact_starts.size > 1:
        for p,c in zip(pre_contact, contact_starts):
            
            if p < 0:
                df['pre_contact'][0:c] = 1
            else:
                df['pre_contact'][p:c] = 1
    else:
        if pre_contact < 0:
            df['pre_contact'][0:int(contact_starts)] = 1
        else:
            df['pre_contact'][int(pre_contact):int(contact_starts)] = 1
            
    
    return df


def get_p_contact_given_approach(df, pre_frames = 15):
    
    if 'pre_contact' not in df.columns:
        df = add_pre_contact(df, pre_frames)
    
    approach_ends = get_approach_end_indices(df)-1
    approach_contact = 0
    
    if approach_ends.size > 1:
        for a in approach_ends:
            if df['pre_contact'][a] == 1:
                approach_contact += 1
    else:
        if df['pre_contact'][approach_ends] == 1:
                approach_contact += 1
    
    p_contact_approach = approach_contact/approach_ends.size
    return p_contact_approach


def get_av_approach_speed(df):
    
    return np.mean(df['mouse_speed'][(df['approach']==1)])


def get_max_approach_speed(df):
    
    return np.max(df['mouse_speed'][(df['approach']==1)])


def get_max_headturn_speed(df, to_capture_time=True, fps=30):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    return fps * np.nanmax(np.abs(df['azimuth_head_arena'][:capture_frame]))


def get_max_bodyturn_speed(df, to_capture_time=True, fps=30):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    return fps * np.nanmax(np.abs(df['azimuth_body_arena'][:capture_frame]))


def plot_hunt(df, to_capture_time=True, video_path=None, save_fig=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    fig, ax = plt.subplots()
    
    plt.plot(df['madj_x'][:capture_frame], df['madj_y'][:capture_frame], c='dodgerblue', zorder=2, label="mouse")
    plt.plot(df['cadj_x'][:capture_frame], df['cadj_y'][:capture_frame], c='k', zorder=1, label="cricket")
    plt.scatter(df['madj_x'][0], df['madj_y'][0], c='cyan', marker='^', zorder=3, label="mouse_start")
    plt.scatter(df['madj_x'][capture_frame-1], df['madj_y'][capture_frame-1], c='cyan', marker='p', zorder=3, label="mouse_end")
    plt.scatter(df['cadj_x'][0], df['cadj_y'][0], c='red', marker='^', zorder=3, label="cricket_start")
    plt.scatter(df['cadj_x'][capture_frame-1], df['cadj_y'][capture_frame-1], c='red', marker='p', zorder=3, label="cricket_end")
    
    if video_path:
        plot_title = video_path.split('/')[-1].split('.')[-2]
        plot_title = plot_title + '_hunt'
    else:
        plot_title = 'hunt'
    
    #plt.xlim((0,45))
    #plt.ylim((0,38))
    plt.title(plot_title)
    #plt.legend(loc="top")
    plt.show()
    
    if save_fig:
        plt.savefig(plot_title+'.png')
    
    return


def plot_approaches(df, to_capture_time=True, video_path=None, save_fig=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    starts = get_approach_start_indices(df, to_capture_time=True)
    ends = get_approach_end_indices(df, to_capture_time=True)
        
    #constrain to be within box
    #draw box
    fig, ax = plt.subplots()
    
    if starts.size == 1:
        plt.plot(df['madj_x'][starts.item():ends.item()], df['madj_y'][starts.item():ends.item()], c='dodgerblue', zorder=2, label="mouse")
   
    elif starts.size > 1:        
        for i in range(starts.size):
            plt.plot(df['madj_x'][starts[i]:ends[i]], df['madj_y'][starts[i]:ends[i]], c='dodgerblue', zorder=2, label="mouse")
            #plt.plot(df['cadj_x'][starts[i]:ends[i]], df['cadj_y'][starts[i]:ends[i]], c='k', zorder=1, label="cricket")
    
    if video_path:
        plot_title = video_path.split('/')[-1].split('.')[-2]
        plot_title = plot_title + '_approaches'
    else:
        plot_title = 'approaches'
    
    #plt.xlim((30,60))
    plt.title(plot_title)
    #plt.legend(loc="top")
    plt.show()
    
    if save_fig:
        plt.savefig(plot_title+'.png')
    
    return
    


def plot_speeds(df, mouse=True, cricket= True, to_capture_time=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    fig, ax = plt.subplots()
    if mouse:
        ax.plot(df['time'][:capture_frame],df['mouse_speed'][:capture_frame], label='mouse')
    if cricket:
        ax.plot(df['time'][:capture_frame],df['cricket_speed'][:capture_frame], label='cricket', color='orange')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('time (s)') 
    plt.ylabel('speed (cm/s)')
    plt.legend()
    plt.show()

    
    return


def plot_distance(df, to_capture_time=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    fig, ax = plt.subplots()
    ax.plot(df['time'][:capture_frame],df['cricket_distance'][:capture_frame])
    plt.xlabel('time (s)') 
    plt.ylabel('distance to cricket (cm)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return

def plot_distance(df, to_capture_time=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    fig, ax = plt.subplots()
    ax.plot(df['time'][:capture_frame],df['cricket_distance'][:capture_frame])
    plt.xlabel('time (s)') 
    plt.ylabel('distance to cricket (cm)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return


def plot_speeds_and_distance(df, mouse=True, cricket= True, contact_distance = 4, to_capture_time=False, show_approaches=True, show_contacts=True, video_path=None, save_fig=False):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    df['mouse_speed'][df['mouse_speed'] <0] = 0
    df['cricket_speed'][df['cricket_speed'] <0] = 0
    df['cricket_distance'][df['cricket_distance'] <0] = 0
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    if mouse:
        ax1.plot(df['time'][:capture_frame],df['mouse_speed'][:capture_frame], label='mouse')
    if cricket:
        ax1.plot(df['time'][:capture_frame],df['cricket_speed'][:capture_frame], label='cricket', color='orange')
        
    ax2.plot(df['time'][:capture_frame],df['cricket_distance'][:capture_frame], color="green", label="cricket distance")
    ax2.axhline(y=contact_distance, linestyle='--', color='r')
    
    if show_approaches:
        ax1.fill_between(df['time'][:capture_frame], 0, df['cricket_speed'].max(), where=df['approach'][:capture_frame] > 0, facecolor='green', alpha=0.3, label='approach')
    if show_contacts: 
        ax1.fill_between(df['time'][:capture_frame], 0, df['cricket_speed'].max(), where=df['contact'][:capture_frame] > 0, facecolor='red', alpha=0.3, label='contact')
    
    
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('speed (cm/s)')
    ax2.set_ylabel('distance to cricket (cm)')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.set_ylim([0,None])
    ax2.set_ylim([0,None])
    ax1.set_xlim([0,None])
    ax2.set_xlim([0,None])
    
    if video_path:
        plot_title = video_path.split('/')[-1].split('.')[-2]
        plot_title = plot_title + '_speed_and_distance'
    else:
        plot_title = 'speed_and_distance'
    
    plt.title(plot_title)
    
    if save_fig:
        plt.savefig(plot_title+'.png')
    
    return


def plot_azimuth_hist(df, n_bins = 20, approach_only=True, video_path=None,save_fig=False):
    
    if approach_only:
        azimuth = df['azimuth_head'][df['approach']==1]
    else:
        azimuth = df['azimuth_head']
    fig, ax = plt.subplots()
        
    ax.hist(azimuth, n_bins, density=True)
    ax.set_xlabel('azimuth (deg)')
    ax.set_ylabel('p (azimuth)')
    ax.set_ylim([0,None])
    ax.set_xlim([-180,180])
    ax.axvline(x=25, color='orange', linestyle='--')
    ax.axvline(x=-25, color='orange', linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if video_path:
        plot_title = video_path.split('/')[-1].split('.')[-2]
        plot_title = plot_title + '_azimtuth_approach_only_' + str(approach_only)
    else:
        plot_title = 'azimtuth_approach_only_' + str(approach_only)
        
    plt.title(plot_title)
#    plt.show(fig) #if you want to run and show outside the script

    if save_fig:
        plt.savefig(plot_title+'.png')
    
    return


def plot_azimuth_polar(df, n_bins = 20, approach_only=True, video_path=None):
    
    if video_path:
        plot_title = video_path.split('/')[-1].split('.')[-2]
        plot_title = plot_title + '_azimtuth_approachOnly_' + str(approach_only)
    else:
        plot_title = 'azimtuth_approachOnly_' + str(approach_only)
    
    if approach_only:
        azimuth = df['azimuth_head'][df['approach']==1]
    else:
        azimuth = df['azimuth_head']
        
    counts, bins = np.histogram(azimuth, bins=n_bins,range=(-180, 180), density=True)
    counts = np.append(counts, counts[0])
    #plot_df = pd.DataFrame(np.array([bins, counts]).T, columns=["bins", "counts"])
    
    plt.polar(np.radians(bins), counts)
    plt.title(plot_title)
    
    return


def plot_azimuth(df,  to_capture_time=False, show_approaches=True, show_contacts=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    fig, ax = plt.subplots()
    
    ax.plot(df['time'][:capture_frame],df['azimuth_head'][:capture_frame], label='azimuth')
    ax.set_ylabel('azimuth (deg)')
    ax.set_xlabel('time (s)')
    ax.set_xlim([0,None])
    ax.set_ylim([-180,180])
    
    if show_approaches:
        ax.fill_between(df['time'][:capture_frame], df['azimuth_head'].min(), df['azimuth_head'].max(), where=df['approach'][:capture_frame] > 0, facecolor='green', alpha=0.3)
    if show_contacts: 
        ax.fill_between(df['time'][:capture_frame], df['azimuth_head'].min(), df['azimuth_head'].max(), where=df['contact'][:capture_frame] > 0, facecolor='red', alpha=0.3)
    
    return


# =============================================================================
# def select_capture_frame_manual(df):
#     
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     ax1.plot(df['frame_number'],df['mouse_speed'], label='mouse')
#     ax1.plot(df['frame_number'],df['cricket_speed'], label='cricket', color='orange')
#     ax2.plot(df['frame_number'],df['cricket_distance'], color="green", label="cricket distance")
#     pts = plt.ginput(1, timeout=-1)
#     capture_frame = pts[0][0]
#     capture_frame = int(round(capture_frame))
#     df['captured'] = 0
#     df['captured'].iloc[capture_frame:] = 1
#     
#     return df
# =============================================================================

def select_capture_frame_manual(df, video_path=0):
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    print("Find capture frame")
    video_scroller(video_path)
    
    capture_frame = int(input("Enter capture frame number: "))
    
    df = set_capture_frame(df, capture_frame)
    print("capture frame set to {}".format(capture_frame))
       
    return df


def set_capture_frame(df, capture_frame):
    
    df['captured'] = 0
    df['captured'].iloc[capture_frame:] = 1
    
    print("captured column added to dataframe")
    
    return df
    

def get_capture_time(df):
    
    return df['time'][df[df['captured']==1].first_valid_index()]


def get_capture_frame(df):
    
    return df[df['captured']==1].first_valid_index()


def get_first_approach_time(df):
    
    return df['time'][df[df['approach']==1].first_valid_index()]


def get_first_contact_time(df):
    
    return df['time'][df[df['contact']==1].first_valid_index()]


def get_number_of_contacts(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    if np.argwhere(np.array(df['contact'][:capture_frame].diff() == 1)).size > 0:
        return len(np.argwhere(np.array(df['contact'][:capture_frame].diff() == 1)))
    else:
        return 1
        
    return 


def get_time_in_contact(df, to_capture_time = True):
    
    return sum(df['contact'][:get_capture_frame(df)])*df['time'].diff().mean()


def get_p_capture_contact(df, to_capture_time=True):
            
    return 1/get_number_of_contacts(df, to_capture_time)


def get_contact_start_indices(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    return np.squeeze(np.argwhere(np.array(df['contact'][:capture_frame].diff() == 1)))


def get_contact_end_indices(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    ends = np.squeeze(np.argwhere(np.array(df['contact'][:capture_frame].diff() == -1)))
    
    if ends.size < get_contact_start_indices(df, to_capture_time).size:
        ends = np.append(ends, get_capture_frame(df))
    
    return ends


def get_number_of_approaches(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    if np.argwhere(np.array(df['approach'][:capture_frame].diff() == 1)).size > 0:
        return len(np.argwhere(np.array(df['approach'][:capture_frame].diff() == 1)))
    else:
        return 1
               
    return 


def get_p_capture_approach(df, to_capture_time=True):
            
    return 1/get_number_of_approaches(df, to_capture_time)


def get_approach_start_indices(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
    
    return np.squeeze(np.argwhere(np.array(df['approach'][:capture_frame].diff() == 1)))


def get_approach_end_indices(df, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
    else:
        capture_frame = df.shape[0]
        
    ends = np.squeeze(np.argwhere(np.array(df['approach'][:capture_frame].diff() == -1)))
    
    if ends.size < get_approach_start_indices(df, to_capture_time).size:
        ends = np.append(ends, get_capture_frame(df))    
    
    return ends


def get_capture_time_relative_to_first_approach_contact(df):
        
    return get_capture_time(df)-min([get_first_contact_time(df),get_first_approach_time(df)])


def select_arena_manual(video_path=0, frame_number = 0):
        
    if not video_path:
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    video = cv2.VideoCapture(video_path)
    
    if not os.path.exists('arena_images'):
        os.makedirs('arena_images')
    
    index = 0
    while(video.isOpened()):
        # Extract images
        ret, frame = video.read()
        # end of frames
        if not ret: 
            break
        # Saves images
        if index == frame_number:
            name = './arena_images/frame' + str(index) + '.jpg'
            print ('Creating...' + name)
            cv2.imwrite(name, frame)
    
        # next frame
        index += 1
    # Read Images 
    img = mpimg.imread(name) 
      
    # Output Images 
    print("select top left, then top right, then bottom left and bottom right corner. Assumes left to right is long side of arena.")
    plt.imshow(img)
    corners = plt.ginput(4, timeout=-1)    
    
    return corners


def pixel_size_from_arena_coordinates(arena_coordinates, arena_size_x = 45, arena_size_y = 38):
    
    left_limit = min([arena_coordinates[0][0], arena_coordinates[2][0]])
    right_limit = max([arena_coordinates[1][0], arena_coordinates[3][0]])
    top_limit = min([arena_coordinates[0][1], arena_coordinates[1][1]])
    bottom_limit = max([arena_coordinates[2][1], arena_coordinates[3][1]])
    
    pixel_x = arena_size_x/(right_limit-left_limit)
    print("pixel x: {}".format(pixel_x))
    
    pixel_y = arena_size_y/(bottom_limit-top_limit)
    print("pixel y: {}".format(pixel_y))
    print("These should be similar")
    
    pixel_size = np.mean([pixel_x, pixel_y])
    
    return pixel_size


def add_corners(df, corners):
    
    corners_x = np.empty((len(df),1))
    corners_x[:] = np.nan
    corners_y = np.empty((len(df),1))
    corners_y[:] = np.nan
    
    for i in range(4):
        corners_x[i] = corners[i][0]
        corners_y[i] = corners[i][1]
    
    df['corners_x'] = corners_x
    df['corners_y'] = corners_y
    
    return df


def get_borders(corners, pts_per_border=1000):
    
    new_order = [0, 1, 3, 2]
    corners = [corners[i] for i in new_order]
    borders = np.zeros((4*pts_per_border,2))
    
    for i in range(4):
        if i < 3:
            borders[i*pts_per_border:(i+1)*pts_per_border,0] = np.linspace(corners[i][0],corners[i+1][0],pts_per_border)
            borders[i*pts_per_border:(i+1)*pts_per_border,1] = np.linspace(corners[i][1],corners[i+1][1],pts_per_border)
        else:
            borders[i*pts_per_border:,0] = np.linspace(corners[i][0],corners[0][0],pts_per_border)
            borders[i*pts_per_border:,1] = np.linspace(corners[i][1],corners[0][1],pts_per_border)
    
    return borders


def affine_transform(video_corners, target_corners, video_points):
    
    trans_matrix = cv2.getPerspectiveTransform(np.float32(video_corners), np.float32(target_corners))
    video_points = np.vstack([video_points, np.ones(video_points.shape[1])])
    target_points = np.matmul(trans_matrix, video_points)
    target_points = np.delete(target_points,obj=2,axis=0)
     
    return target_points


def get_distribution(df, dist_bins, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
        df = df.loc[:capture_frame,:]
    
    mouse, bin_edges = np.histogram(df['mouse_to_borders'][(df['approach']==0) & (df['contact']==0)], bins=dist_bins, density=True)
    cricket, bin_edges = np.histogram(df['cricket_to_borders'][(df['approach']==0) & (df['contact']==0)], bins=dist_bins, density=True)
    mouse_approach, bin_edges = np.histogram(df['mouse_to_borders'][df['approach']==1], bins=dist_bins, density=True)
    cricket_approach, bin_edges = np.histogram(df['cricket_to_borders'][df['approach']==1], bins=dist_bins, density=True)
    mouse_contact, bin_edges = np.histogram(df['mouse_to_borders'][df['contact']==1], bins=dist_bins, density=True)
    cricket_contact, bin_edges = np.histogram(df['cricket_to_borders'][df['contact']==1], bins=dist_bins, density=True)
    mouse_approach_contact, bin_edges = np.histogram(df['mouse_to_borders'][(df['approach']==1) | (df['contact']==1)], bins=dist_bins, density=True)
    cricket_approach_contact, bin_edges = np.histogram(df['cricket_to_borders'][(df['approach']==1) | (df['contact']==1)], bins=dist_bins, density=True)
    
    dist_dict = {'dist_to_border': dist_bins[:-1]+(np.diff(dist_bins)/2),
                 'mouse': mouse,
                 'cricket': cricket,
                 'mouse_approach': mouse_approach,
                 'cricket_approach': cricket_approach,
                 'mouse_contact': mouse_contact,
                 'cricket_contact': cricket_contact,
                 'mouse_approach_contact': mouse_approach_contact,
                 'cricket_approach_contact': cricket_approach_contact}
    
    df_distribution = pd.DataFrame(dist_dict)
    
    return df_distribution


def get_density(df, bin_num, bin_range, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
        df = df.loc[:capture_frame,:]
    
    min_x = np.min(df['madj_x'])
    min_y = np.min(df['madj_y'])
    
    if min_x < 0:
        df['madj_x'] = df['madj_x'] - min_x
        df['cadj_x'] = df['cadj_x'] - min_x
        
    if min_y < 0:
        df['madj_y'] = df['madj_y'] - min_y
        df['cadj_y'] = df['cadj_y'] - min_y
    
    max_x = np.max(df['madj_x'])
    max_y = np.max(df['madj_y'])
    
    if max_x > bin_range[0][1]:
        df['madj_x'] = df['madj_x'] / max_x * bin_range[0][1]
        df['cadj_x'] = df['cadj_x'] / max_x * bin_range[0][1]
        
    if max_y > bin_range[1][1]:
        df['madj_y'] = df['madj_y'] / max_y * bin_range[1][1]
        df['cadj_y'] = df['cadj_y'] / max_y * bin_range[1][1]
        
    mouse, xedges, yedges = np.histogram2d(df['madj_x'], df['madj_y'], bins=bin_num, range=bin_range, density=True)
    mouse = np.reshape(np.transpose(mouse), mouse.shape[0] * mouse.shape[1])
    cricket, xedges, yedges = np.histogram2d(df['cadj_x'], df['cadj_y'], bins=bin_num, range=bin_range, density=True)
    cricket = np.reshape(np.transpose(cricket), cricket.shape[0] * cricket.shape[1])
    mouse_approach, xedges, yedges = np.histogram2d(df['madj_x'][df['approach']==1], df['madj_y'][df['approach']==1], bins=bin_num, range=bin_range, density=True)
    mouse_approach = np.reshape(np.transpose(mouse_approach), mouse_approach.shape[0] * mouse_approach.shape[1])
    cricket_approach, xedges, yedges = np.histogram2d(df['cadj_x'][df['approach']==1], df['cadj_y'][df['approach']==1], bins=bin_num, range=bin_range, density=True)
    cricket_approach = np.reshape(np.transpose(cricket_approach), cricket_approach.shape[0] * cricket_approach.shape[1])
    mouse_contact, xedges, yedges = np.histogram2d(df['madj_x'][df['contact']==1], df['madj_y'][df['contact']==1], bins=bin_num, range=bin_range, density=True)
    mouse_contact = np.reshape(np.transpose(mouse_contact), mouse_contact.shape[0] * mouse_contact.shape[1])
    cricket_contact, xedges, yedges = np.histogram2d(df['cadj_x'][df['contact']==1], df['cadj_y'][df['contact']==1], bins=bin_num, range=bin_range, density=True)
    cricket_contact = np.reshape(np.transpose(cricket_contact), cricket_contact.shape[0] * cricket_contact.shape[1])
    mouse_approach_contact, xedges, yedges = np.histogram2d(df['madj_x'][(df['approach']==1) | (df['contact']==1)], df['madj_y'][(df['approach']==1) | (df['contact']==1)], bins=bin_num, range=bin_range, density=True)
    mouse_approach_contact = np.reshape(np.transpose(mouse_approach_contact), mouse_approach_contact.shape[0] * mouse_approach_contact.shape[1])
    cricket_approach_contact, xedges, yedges = np.histogram2d(df['cadj_x'][(df['approach']==1) | (df['contact']==1)], df['cadj_y'][(df['approach']==1) | (df['contact']==1)], bins=bin_num, range=bin_range, density=True)
    cricket_approach_contact = np.reshape(np.transpose(cricket_approach_contact), cricket_approach_contact.shape[0] * cricket_approach_contact.shape[1])
    
    dens_dict = {'mouse': mouse,
                 'cricket': cricket,
                 'mouse_approach': mouse_approach,
                 'cricket_approach': cricket_approach,
                 'mouse_contact': mouse_contact,
                 'cricket_contact': cricket_contact,
                 'mouse_approach_contact': mouse_approach_contact,
                 'cricket_approach_contact': cricket_approach_contact}
    
    df_density = pd.DataFrame(dens_dict)
    
    return df_density


def get_azimuth_hist(df, azimuth_bins, max_dist_azimuth, to_capture_time=True):
    
    if to_capture_time:
        capture_frame = get_capture_frame(df)
        df = df.loc[:capture_frame,:]
    
    azimuth_head_explore, bin_edges = np.histogram(df['azimuth_head'][(df['approach']==0) & (df['contact']==0)], bins=azimuth_bins, density=True)
    azimuth_body_explore, bin_edges = np.histogram(df['azimuth_body'][(df['approach']==0) & (df['contact']==0)], bins=azimuth_bins, density=True)
    azimuth_head_approach, bin_edges = np.histogram(df['azimuth_head'][df['approach']==1], bins=azimuth_bins, density=True)
    azimuth_body_approach, bin_edges = np.histogram(df['azimuth_body'][df['approach']==1], bins=azimuth_bins, density=True)
    azimuth_head_contact, bin_edges = np.histogram(df['azimuth_head'][df['contact']==1], bins=azimuth_bins, density=True)
    azimuth_body_contact, bin_edges = np.histogram(df['azimuth_body'][df['contact']==1], bins=azimuth_bins, density=True)
    azimuth_head_approach_contact, bin_edges = np.histogram(df['azimuth_head'][(df['approach']==1) | (df['contact']==1)], bins=azimuth_bins, density=True)
    azimuth_body_approach_contact, bin_edges = np.histogram(df['azimuth_body'][(df['approach']==1) | (df['contact']==1)], bins=azimuth_bins, density=True)
    azimuth_head_max_dist, bin_edges = np.histogram(df['azimuth_head'][(df['approach']==1) | (df['contact']==1) & (df['cricket_distance']<=max_dist_azimuth)], bins=azimuth_bins, density=True)
    azimuth_body_max_dist, bin_edges = np.histogram(df['azimuth_body'][(df['approach']==1) | (df['contact']==1) & (df['cricket_distance']<=max_dist_azimuth)], bins=azimuth_bins, density=True)
    
    azimuth_dict = {'azimuth_angle': azimuth_bins[:-1]+(np.diff(azimuth_bins)/2),
                 'azimuth_head_explore': azimuth_head_explore,
                 'azimuth_body_explore': azimuth_body_explore,
                 'azimuth_head_approach': azimuth_head_approach,
                 'azimuth_body_approach': azimuth_body_approach,
                 'azimuth_head_contact': azimuth_head_contact,
                 'azimuth_body_contact': azimuth_body_contact,
                 'azimuth_head_approach_contact': azimuth_head_approach_contact,
                 'azimuth_body_approach_contact': azimuth_body_approach_contact,
                 'azimuth_head_max_dist': azimuth_head_max_dist,
                 'azimuth_body_max_dist': azimuth_body_max_dist}
    
    df_azimuth = pd.DataFrame(azimuth_dict)
    
    return df_azimuth


def get_fps_from_video(video_path = 0):
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)    
    video.release()    
    fps = int(round(fps))    
    
    return fps


def annotate_video(df, video_path = 0, fps_out=30, to_capture_time=True, show_time=True, show_borders=True, label_bodyparts=True, show_approaches=True, show_approach_number=True, show_contacts=True, show_contact_number=True, show_azimuth=True, show_azimuth_lines=True, show_distance=True, show_speed=True, save_video=True, show_video=True, border_size=40, video_path_ext=''):

    #add circle to contacts
    #add trail option
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    if save_video:
        print("original video path: " + video_path)
        save_path = video_path.split('/')[-1].split('.')[-2] + video_path_ext + ".avi"
        #save_filename = input("Enter save filename without .avi: ")
        #save_path = os.path.split(video_path)[0]+'/'+save_filename+'.avi'


    #load video
    video = cv2.VideoCapture(video_path)
    
    #get video info
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    #set output video
    #output_video_filename = "video_3.avi"
    #video_out = cv2.VideoWriter(output_video_filename,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width+2*border_size,frame_height+2*border_size))
    
    #set window size for video with border
    cv2.namedWindow('bordered_video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bordered_video', frame_width+2*border_size,frame_height+2*border_size)
    
    if save_video:
        video_out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps_out, (frame_width+2*border_size,frame_height+2*border_size))

    
    video_duration = n_frames/fps
    time = np.linspace(0, video_duration, n_frames)
    count = 0
    
    #os.chdir(video_path)
    
    contact_index = get_contact_start_indices(df, to_capture_time=to_capture_time)
    approach_index = get_approach_start_indices(df, to_capture_time=to_capture_time)
    contact_index_count = 0
    approach_index_count = 0
    while(video.isOpened()):
      # Capture frame-by-frame
      ret, frame = video.read()
      if ret == True:
         
        if label_bodyparts:
            #cv2.circle(frame, (int(df['mid_x'][count]),int(df['mid_y'][count])), 1, (0,255,0), thickness=1)
            cv2.circle(frame, (int(df['rightear_x'][count]),int(df['rightear_y'][count])), 2, (0,0,255), thickness=1)
            cv2.circle(frame, (int(df['leftear_x'][count]),int(df['leftear_y'][count])), 2, (0,0,255), thickness=1)
            cv2.circle(frame, (int(df['nose_x'][count]),int(df['nose_y'][count])), 2, (0,0,255), thickness=1)
            cv2.circle(frame, (int(df['tailbase_x'][count]),int(df['tailbase_y'][count])), 2, (0,0,255), thickness=1)
            cv2.circle(frame, (int(df['cricket_x'][count]),int(df['cricket_y'][count])), 2, (0,0,255), thickness=1)
            cv2.line(frame,(int(df['mid_x'][count]),int(df['mid_y'][count])),(int(df['tailbase_x'][count]),int(df['tailbase_y'][count])),(0,255,0,),thickness=1, lineType=8)
            cv2.line(frame,(int(df['cricket_x'][count]),int(df['cricket_y'][count])),(int(df['tailbase_x'][count]),int(df['tailbase_y'][count])),(0,255,0,),thickness=1, lineType=8)
            #cv2.circle(frame, (int(df['head_x'][count]),int(df['head_y'][count])), 1, (0,255,0), thickness=1)
            #continue
        
        if show_azimuth_lines:
            if df['cricket_likelihood'][count] >= 0.9:
                #cv2.line(frame, (int(df['leftear_x'][count]),int(df['leftear_y'][count])), (int(df['cricket_x'][count]),int(df['cricket_y'][count])), (255, 0, 0), thickness=1, lineType=8)
                #cv2.line(frame, (int(df['rightear_x'][count]),int(df['rightear_y'][count])), (int(df['cricket_x'][count]), int(df['cricket_y'][count])), (255, 0, 0), thickness=1, lineType=8)
                cv2.line(frame, (int(df['mid_x'][count]),int(df['mid_y'][count])), (int(df['cricket_x'][count]), int(df['cricket_y'][count])), (0, 255, 0), thickness=1, lineType=8)
                #cv2.line(frame, (int(df['head_x'][count]),int(df['head_y'][count])), (int(df['cricket_x'][count]), int(df['cricket_y'][count])), (255, 0, 0), thickness=1, lineType=8)
            #else:
                #cv2.line(frame, (int(df['leftear_x'][count]),int(df['leftear_y'][count])), (int(df['mid_x'][count]),int(df['mid_y'][count])), (255, 0, 0), thickness=1, lineType=8)
                #cv2.line(frame, (int(df['rightear_x'][count]),int(df['rightear_y'][count])), (int(df['mid_x'][count]),int(df['mid_y'][count])), (255, 0, 0), thickness=1, lineType=8)
        
        if show_borders:
            corners = np.array([(df['corners_x'][0],df['corners_y'][0]), (df['corners_x'][1],df['corners_y'][1]), (df['corners_x'][3], df['corners_y'][3]), (df['corners_x'][2],df['corners_y'][2])])
            corners = corners.astype(int)
            for i in range(4):
                if i < 3:
                    cv2.line(frame, (corners[i][0],corners[i][1]), (corners[i+1][0],corners[i+1][1]), (255,0,0), thickness=1, lineType=8)
                else:
                    cv2.line(frame, (corners[i][0],corners[i][1]), (corners[0][0],corners[0][1]), (255,0,0), thickness=1, lineType=8)
               
        bordered_video = cv2.copyMakeBorder(frame, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType=cv2.BORDER_CONSTANT, value=0)
        
        overlay_image = np.copy(bordered_video)
        
        if show_time:
            cv2.putText(img=overlay_image, text = "time: "+ str(round(time[count], 2)), org=(int(20),int(border_size/2)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
        
        if show_approaches and df['approach'][count]:
            cv2.putText(img=overlay_image, text = "approach", org=(int(140),int(border_size/2)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 0, 255], thickness=1)
        
        if show_approach_number:
            if approach_index_count < approach_index.size and approach_index.size > 1:
                if count == approach_index[approach_index_count]:
                    approach_index_count = approach_index_count + 1
            else:
                if approach_index.size > 0:
                    if count == approach_index.item(0):
                         approach_index_count = approach_index_count + 1
            cv2.putText(img=overlay_image, text = "# approaches: "+ str(approach_index_count), org=(int(140),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 0, 255], thickness=1)
    
            
        if show_contacts and df['contact'][count]:
            cv2.putText(img=overlay_image, text = "contact", org=(int(260),int(border_size/2)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[0, 255, 255], thickness=1)
            
        if show_contact_number:
            if contact_index_count < contact_index.size and contact_index.size > 1:
                if count == contact_index[contact_index_count]:
                    contact_index_count = contact_index_count + 1
            else:
                if contact_index.size > 0:
                    if count == contact_index.item(0):
                        contact_index_count = contact_index_count + 1                                        
            cv2.putText(img=overlay_image, text = "# contacts: "+ str(contact_index_count), org=(int(260),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[0, 255, 255], thickness=1)
            #if contact_index_count >= len(contact_index):
                #contact_index_count = contact_index_count-1
            
        if show_azimuth:
            #cv2.putText(img=overlay_image, text = "azimuth: "+ str(round(df['azimuth_head'][count])), org=(int(20),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
            cv2.putText(img=overlay_image, text = "azimuth: "+ str(round(df['azimuth_body'][count])), org=(int(20),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
            #cv2.putText(img=overlay_image, text = "distance: "+ str(round(df['mouse_to_borders'][count],2)), org=(int(20),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
        
        if show_distance:
            #cv2.putText(img=overlay_image, text = "distance (cm): "+ str(round(df['cricket_distance'][count],2)), org=(int(520),int(frame_height++1.5*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
            cv2.putText(img=overlay_image, text = "to borders (cm): "+ str(round(df['mouse_to_borders'][count],2)), org=(int(20),int(frame_height++0.75*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
        
        if show_speed:
            cv2.putText(img=overlay_image, text = "speed (cm/s): "+ str(round(df['mouse_speed'][count],2)), org=(int(220),int(frame_height++0.75*border_size)),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[255, 255, 255], thickness=1)
        
            
            
        cv2.addWeighted(src1=overlay_image, alpha=0.99, src2=bordered_video, beta=0.5, gamma=0, dst=bordered_video)
        
        if show_video:
            cv2.imshow('bordered_video', bordered_video)
            #cv2.waitKey(int((1/fps_out)*1000))
        
        if save_video:
            video_out.write(bordered_video)
    
        count = count+1
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break
     
    # When everything done, release the video capture object
    video.release()
    if save_video:
        video_out.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
    
    return


def video_scroller(video_path = 0):
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    
    def onChange(trackbarValue):
        video.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        err,img = video.read()
        cv2.imshow("mywindow", img)
        pass
    
    cv2.namedWindow('mywindow')
    cv2.createTrackbar( 'frame', 'mywindow', 0, length, onChange )
    cv2.createTrackbar( 'end_frame'  , 'mywindow', length, length, onChange )
    
    
    onChange(0)
    cv2.waitKey()
    
    start = cv2.getTrackbarPos('start','mywindow')
    end   = cv2.getTrackbarPos('end','mywindow')
    
    video.set(cv2.CAP_PROP_POS_FRAMES,start)
    while video.isOpened():
        err,img = video.read()
        if video.get(cv2.CAP_PROP_POS_FRAMES) >= end:
            break
        cv2.imshow("mywindow", img)
        k = cv2.waitKey(10) & 0xff
        if k==27:
            break

    
    
    return


def get_save_path_csv(h5_file):
         
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_pythonAnalysis.csv'
 

def get_save_path_csv_summary(h5_file):
        
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_pythonAnalysis_summary.csv'


def get_save_path_csv_distribution(h5_file):
    
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_distribution.csv'


def get_save_path_csv_density(h5_file):
    
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_density.csv'


def get_save_path_csv_azimuth(h5_file):
    
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_azimuth.csv'


def get_save_path_video(h5_file):
            
    return os.path.basename(os.path.splitext(h5_file)[0]) +'_annotatedVideo.avi'
    

class Window(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.lbl = QLabel('select part to label')
        self.cricket = QRadioButton('cricket')
        self.leftear = QRadioButton('leftear')
        self.nose = QRadioButton('nose')
        self.rightear = QRadioButton('rightear')
        self.tailbase = QRadioButton('tailbase')
        #self.btn = QPushButton('Select')

        layout = QVBoxLayout()
        layout.addWidget(self.lbl)
        layout.addWidget(self.cricket)
        layout.addWidget(self.leftear)
        layout.addWidget(self.nose)
        layout.addWidget(self.rightear)
        layout.addWidget(self.tailbase)
        #layout.addWidget(self.btn)
        

        self.setLayout(layout)
        self.setWindowTitle('select bodypart to label')
        self.cricket.setChecked(True)
        self.show()
        

def check_buttons(button_window):
    if button_window.cricket.isChecked():
        return 'cricket_likelihood','cricket_x', 'cricket_y'
    elif button_window.leftear.isChecked():
        return 'leftear_likelihood','leftear_x', 'leftear_y'
    elif button_window.nose.isChecked():
        return 'nose_likelihood','nose_x', 'nose_y'
    elif button_window.rightear.isChecked():
        return 'rightear_likelihood','rightear_x', 'rightear_y'
    else:
        return 'tailbase_likelihood','tailbase_x', 'tailbase_y'
    

def adjust_label_positions(df, video_path = 0):
    
    global length
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    app = QApplication(sys.argv)
    button_window = Window()
    label_likelihood, label_x, label_y = check_buttons(button_window)
    
    video = cv2.VideoCapture(video_path)
    
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))-1
    
    
    def onChange(trackbarValue):
        label_likelihood, label_x, label_y = check_buttons(button_window)
        frame = cv2.getTrackbarPos('frame','mywindow')
        video.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        err,img = video.read()
        #cv2.putText(img=img, text = "*", org=(int(df[label_x][cv2.getTrackbarPos('frame','mywindow')]),int(df[label_y][cv2.getTrackbarPos('frame','mywindow')])),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[0, 0, 255], thickness=1)
        cv2.drawMarker(img=img, position=(int(df[label_x][frame]),int(df[label_y][frame])), color=[0, 0, 255], thickness=1)
        cv2.putText(img=img, text = str(round(df[label_likelihood][cv2.getTrackbarPos('frame','mywindow')],2)), org=(5,25),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[0, 0, 255], thickness=1)
        cv2.imshow("mywindow", img)
        pass
    
    def change_labels(f, X, Y, label_x, label_y, df=df):
        
        df[label_x][f] = X
        df[label_y][f] = Y
        
        return
    
    def move_trackbar(value):
        frame = cv2.getTrackbarPos('frame','mywindow')
        if value < 0 and frame == 0:
            return
        elif value > 0 and frame == length:
            return
        else:
            cv2.setTrackbarPos('frame', 'mywindow', frame+value)
        
        return
    
    def onMouse(event, x, y, flags, param):
        #global labels_to_change
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                move_trackbar(1)
            elif flags > 0:
                move_trackbar(-1)
            else:
                pass
            
        if event == cv2.EVENT_LBUTTONDOWN:
            label_likelihood, label_x, label_y = check_buttons(button_window)
            frame = cv2.getTrackbarPos('frame','mywindow')
            print('frame = %d'%(frame))
            print('x = %d, y = %d'%(x, y))
            change_labels(frame, x, y, label_x, label_y)
            move_trackbar(1)
        return
             
    
    cv2.namedWindow('mywindow',0)
    cv2.createTrackbar( 'frame', 'mywindow', 0, length, onChange )
    cv2.setMouseCallback('mywindow', onMouse)
    #cv2.createTrackbar( 'end_frame'  , 'mywindow', length, length, onChange )
    
    
    #onChange(0)
    
    
    frame = cv2.getTrackbarPos('frame','mywindow')
    #end   = cv2.getTrackbarPos('end','mywindow')
    
    
    video.set(cv2.CAP_PROP_POS_FRAMES,frame)
    err,img = video.read()
    label_likelihood, label_x, label_y = check_buttons(button_window)
    cv2.drawMarker(img=img, position=(int(df[label_x][frame]),int(df[label_y][frame])), color=[0, 0, 255], thickness=1)
    #cv2.putText(img=img, text = "*", org=(int(df[label_x][cv2.getTrackbarPos('frame','mywindow')]),int(df[label_y][cv2.getTrackbarPos('frame','mywindow')])),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=.4, color=[0, 0, 255], thickness=1)
    cv2.imshow("mywindow", img)
    cv2.waitKey()

    while video.isOpened():
        cv2.imshow("mywindow", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            button_window.close()
            break  

    
    cv2.destroyAllWindows()
    cv2.waitKey(0)
    return df


def slowmo(video_path = 0, save_video=False, show_video=True):
    
    if not video_path:
        print("select video file")
        root = Tk()
        video_path = filedialog.askopenfilenames(parent=root,title='Choose files')
        video_path = video_path[0]
        
    if save_video:
        print("original video path: " + video_path)
        save_filename = input("Enter save filename without .avi: ")
        save_path = os.path.split(video_path)[0]+'/'+save_filename+'.avi'

        
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #output_video_filename = "121319_1w1_2_day5__2_azimuth_slow.avi"
    video_out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 10,(frame_width,frame_height))

    
    #video.set(cv2.CAP_PROP_FPS,10)
    while(video.isOpened()):
      # Capture frame-by-frame
      ret, frame = video.read()
      if ret == True:
          
          if show_video:
              cv2.imshow('video_slow', frame)
              cv2.waitKey(100)
              
          
          if save_video:
            video_out.write(frame)
      else:
          break

    video.release()
    video_out.release()

    cv2.destroyAllWindows()
    
    return


def get_trial_id(video_path):
    
    base=os.path.basename(video_path)
    trial_id = os.path.splitext(base)[0]
    print(trial_id)
    
    return trial_id


def get_save_path(video_path):
    
    save_path = os.path.dirname(os.path.realpath(video_path))
    
    return save_path


def save_summary_df(df, video_path):
    
    save_path = get_save_path(video_path)
    
    save_name = save_path + "\\" + get_trial_id(video_path) + "_summary.csv"
    
    df.to_csv(save_name, index=False)
    
    print("saved dataframe as: {}".format(save_name))
    
    return


def save_analysis_df(df, video_path):
    
    save_path = get_save_path(video_path)
    
    save_name = save_path + "\\" + get_trial_id(video_path) + "_analysis.csv"
    
    df.to_csv(save_name, index=False)
    
    print("saved dataframe as: {}".format(save_name))
    
    return
    
    
def summarize_df(df, trial_id=None, condition=None):
    
    summary_df = pd.DataFrame()
    summary_df = summary_df.append({'trial_id': trial_id}, ignore_index=True)
    summary_df['condition'] = condition
    summary_df['capture_time'] = get_capture_time(df)
    summary_df['capture_time_first_approach_contact'] = get_capture_time_relative_to_first_approach_contact(df)
    summary_df['number_of_approaches'] = get_number_of_approaches(df, to_capture_time=True)
    summary_df['p_contact_approach'] = get_p_contact_given_approach(df, pre_frames=15)
    summary_df['p_capture_approach'] = get_p_capture_approach(df, to_capture_time=True)
    summary_df['first_approach_latency'] = get_first_approach_time(df)
    
    summary_approach_intervals = get_approach_intervals(df, to_capture_time=True, fps=30)
    if summary_approach_intervals.size > 1:
        summary_df['median_approach_interval'] = summary_approach_intervals[0]
        summary_df['mean_approach_interval'] = summary_approach_intervals[1]
    else:
        summary_df['median_approach_interval'] = 0
        summary_df['mean_approach_interval'] = 0
    
    summary_df['median_approach_to_border'], summary_df['mean_approach_to_border'] = get_approach_path_to_border(df, to_capture_time=True)
   # summary_df['interapproach_interval_median'] = get_median_interapproach_interval(df, to_capture_time=True, fps = 30)
   # summary_df['interapproach_interval_mean'] = get_mean_interapproach_interval(df, to_capture_time=True, fps = 30)
    summary_df['number_of_contacts'] = get_number_of_contacts(df, to_capture_time=True)
    summary_df['p_capture_contact'] = get_p_capture_contact(df, to_capture_time=True)
    summary_df['time_in_contact'] = get_time_in_contact(df, to_capture_time = True)
    #summary_df['intercontact_interval_median'] = get_median_intercontact_interval(df, to_capture_time=True, fps = 30)
    #summary_df['intercontact_interval_mean'] = get_mean_intercontact_interval(df, to_capture_time=True, fps = 30)
    summary_df['first_contact_latency'] = get_first_contact_time(df) 
    summary_df['approach_speed_av'] = get_av_approach_speed(df)
    summary_df['approach_speed_max'] = get_max_approach_speed(df)
    summary_df['headturn_speed_max'] = get_max_headturn_speed(df, to_capture_time=True, fps=30)
    summary_df['bodyturn_speed_max'] = get_max_bodyturn_speed(df, to_capture_time=True, fps=30)

    return summary_df


def concatenate_dfs(df_paths = 0):
    
    if not df_paths:
        print("select dfs to concatenate")
        root = Tk()
        df_paths = filedialog.askopenfilenames(parent=root,title='Choose files')
    
    first = 1
    for df in df_paths:
        if first:
            print('vertically stacking: ')
            print(df)
            concatenated_df = pd.read_csv(df)
            first = first-1
        else:
            print(df)
            current_df = pd.read_csv(df)
            concatenated_df = pd.concat([concatenated_df, current_df], ignore_index=True)
    
    if 'Unnamed: 0' in concatenated_df.columns:
        concatenated_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return concatenated_df













