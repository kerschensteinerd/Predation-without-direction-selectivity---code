# -*- coding: utf-8 -*-
"""
Hunting Behavior Analysis Script

Main script for analyzing mouse hunting behavior from video pose tracking.
Processes video files, DeepLabCut tracking data, and generates comprehensive
behavioral analysis including approach detection, contact events, and spatial distributions.

Author: KerschensteinerLab
Created: Wed Aug 26 20:23:40 2020
Updated: 2024 - Added CLI support, error handling, and configuration management
"""

import sys
import argparse
import pandas as pd
import numpy as np
import tkinter
from tkinter import filedialog, Tk
import os
import matplotlib.pyplot as plt
from hunting_analysis_functions import *
from config import AnalysisConfig


def parse_arguments():
    """Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze mouse hunting behavior from video tracking data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (GUI file selection):
  python hunting_analysis_script.py
  
  # Command-line mode with all inputs:
  python hunting_analysis_script.py \\
      --video trial_001.mp4 \\
      --h5 trial_001DLC.h5 \\
      --csv trial_001_Analysis.csv \\
      --capture-frame 1500 \\
      --output-dir ./results
        """
    )
    
    # File inputs
    parser.add_argument('--video', type=str, help='Path to .mp4 video file')
    parser.add_argument('--h5', type=str, help='Path to .h5 pose tracking file')
    parser.add_argument('--csv', type=str, help='Path to _Analysis.csv file')
    
    # Analysis parameters
    parser.add_argument('--capture-frame', type=int, help='Frame number when prey was captured')
    parser.add_argument('--output-dir', type=str, help='Directory for output files (default: same as video)')
    parser.add_argument('--condition', type=str, default='WT', help='Experimental condition label (default: WT)')
    
    # Configuration overrides
    parser.add_argument('--frame-rate', type=float, help='Video frame rate (fps)')
    parser.add_argument('--arena-width', type=float, help='Arena width (cm)')
    parser.add_argument('--arena-height', type=float, help='Arena height (cm)')
    parser.add_argument('--speed-threshold', type=float, help='Minimum speed for approach (cm/s)')
    parser.add_argument('--contact-distance', type=float, help='Contact detection distance (cm)')
    
    # Output options
    parser.add_argument('--no-video', action='store_true', help='Skip annotated video generation')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    return parser.parse_args()


def get_file_paths_interactive():
    """Get file paths using interactive GUI dialogs.
    
    Returns
    -------
    tuple
        (video_path, h5_file, df_path) file paths
    """
    root = Tk()
    root.withdraw()  # Hide main window
    
    print("Select files using dialogs...")
    video_path = filedialog.askopenfilename(parent=root, title='Choose .mp4 file')
    if not video_path:
        root.destroy()
        sys.exit("No video file selected. Exiting.")
    
    h5_file = filedialog.askopenfilename(parent=root, title='Choose .h5 file')
    if not h5_file:
        root.destroy()
        sys.exit("No .h5 file selected. Exiting.")
    
    df_path = filedialog.askopenfilename(parent=root, title='Choose _Analysis .csv file')
    if not df_path:
        root.destroy()
        sys.exit("No CSV file selected. Exiting.")
    
    root.destroy()
    return video_path, h5_file, df_path


def load_dataframe(df_path):
    """Load and preprocess tracking dataframe.
    
    Parameters
    ----------
    df_path : str
        Path to CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded tracking data
        
    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    pd.errors.EmptyDataError
        If file is empty
    """
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Analysis file not found: {df_path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Analysis file is empty: {df_path}")
    
    # Remove pandas default index column if present
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return df


def get_capture_frame_interactive():
    """Get capture frame number from user input.
    
    Returns
    -------
    int
        Frame number when prey was captured
    """
    while True:
        try:
            capture_frame = int(input("Enter capture frame number: "))
            if capture_frame < 0:
                print("Error: Capture frame must be non-negative. Try again.")
                continue
            return capture_frame
        except ValueError:
            print("Error: Please enter a valid integer. Try again.")
        except KeyboardInterrupt:
            print("\nAborted by user.")
            sys.exit(1)


def main():
    """Main analysis workflow."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = AnalysisConfig()
    
    # Override config with command-line arguments if provided
    if args.frame_rate is not None:
        config.frame_rate = args.frame_rate
    if args.arena_width is not None:
        config.arena_width = args.arena_width
    if args.arena_height is not None:
        config.arena_height = args.arena_height
    if args.speed_threshold is not None:
        config.speed_threshold = args.speed_threshold
    if args.contact_distance is not None:
        config.contact_distance = args.contact_distance
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Get file paths (CLI or interactive)
    if args.video and args.h5 and args.csv:
        video_path = args.video
        h5_file = args.h5
        df_path = args.csv
        print(f"Using files from command line:")
        print(f"  Video: {video_path}")
        print(f"  H5: {h5_file}")
        print(f"  CSV: {df_path}")
    else:
        video_path, h5_file, df_path = get_file_paths_interactive()
    
    # Verify files exist
    for path, name in [(video_path, 'Video'), (h5_file, 'H5'), (df_path, 'CSV')]:
        if not os.path.exists(path):
            sys.exit(f"Error: {name} file not found: {path}")
    
    # Load dataframe
    print("\nLoading tracking data...")
    try:
        df = load_dataframe(df_path)
        print(f"✓ Loaded {len(df)} frames")
    except Exception as e:
        sys.exit(f"Error loading CSV: {e}")
    
    # Set working directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    else:
        os.chdir(os.path.dirname(os.path.abspath(video_path)))
    
    # Calibrate arena
    print("\nCalibrating arena...")
    try:
        corners = select_arena_manual(video_path, frame_number=0)
        cm_per_pixel = pixel_size_from_arena_coordinates(
            corners, 
            arena_size_x=config.arena_width, 
            arena_size_y=config.arena_height
        )
        print(f"✓ Calibration: {cm_per_pixel:.4f} cm/pixel")
    except Exception as e:
        sys.exit(f"Error during calibration: {e}")
    
    # Get capture frame
    if args.capture_frame is not None:
        capture_frame = args.capture_frame
        print(f"Using capture frame from command line: {capture_frame}")
    else:
        capture_frame = get_capture_frame_interactive()
    
    # ==================== Compute Analysis Features ====================
    print("\nComputing behavioral features...")
    
    df = set_capture_frame(df, capture_frame)
    df = calculate_head(df)
    df = get_azimuth_head(df)
    df = get_azimuth_body(df)
    df = get_distance_to_cricket(df, cm_per_pixel)
    df = get_mouse_speed(df, config.frame_rate, cm_per_pixel, 
                         smooth_frames=config.smooth_frames, 
                         smooth_order=config.smooth_order)
    df = get_cricket_speed(df, config.frame_rate, cm_per_pixel, 
                           smooth_frames=config.smooth_frames, 
                           smooth_order=config.smooth_order)
    df = get_mouse_acceleration(df, smooth_frames=config.acceleration_smooth_frames, 
                                smooth_order=config.smooth_order)
    df = get_contacts(df, config.contact_distance)
    df = smooth_contacts(df, window_size=int(config.window_size//2))
    df = get_approaches(df, config.speed_threshold, 
                       diff_frames=config.diff_frames, 
                       diff_speed=config.diff_speed, 
                       frame_rate=config.frame_rate, 
                       body_azimuth=config.body_azimuth_threshold)
    df = smooth_approaches(df, window_size=config.window_size)
    
    # No approach while contact
    df['approach'][df['contact'] == 1] = 0
    
    print("✓ Behavioral features computed")
    
    # ==================== Spatial Transformations ====================
    print("\nApplying spatial transformations...")
    
    target_corners = np.array([
        [0, 0], 
        [config.arena_width, 0], 
        [0, config.arena_height], 
        [config.arena_width, config.arena_height]
    ])
    
    mouse_points = np.array((df['mid_x'], df['mid_y']))
    mouse_adjusted = affine_transform(corners, target_corners, mouse_points)
    df[['madj_x', 'madj_y']] = pd.DataFrame(np.transpose(mouse_adjusted))
    
    cricket_points = np.array((df['cricket_x'], df['cricket_y']))
    cricket_adjusted = affine_transform(corners, target_corners, cricket_points)
    df[['cadj_x', 'cadj_y']] = pd.DataFrame(np.transpose(cricket_adjusted))
    
    df = get_azimuth_head_arena(df)
    df = get_azimuth_body_arena(df)
    
    borders = get_borders(target_corners, pts_per_border=1000)
    df = get_distance_to_borders(df, borders)
    df = get_distance_path_to_borders(df, borders, n_samples=100)
    df = add_corners(df, corners)
    
    print("✓ Spatial transformations applied")
    
    # ==================== Compute Distributions ====================
    print("\nComputing distributions...")
    
    dist_bins = np.linspace(0, config.max_border_distance, 
                           round(config.max_border_distance/config.bin_size)+1)
    df_distribution = get_distribution(df, dist_bins, to_capture_time=True)
    
    df_density = get_density(df, config.bin_num, config.get_bin_range(), to_capture_time=True)
    
    azimuth_bins = np.linspace(
        config.azimuth_range[0], 
        config.azimuth_range[1],
        int(np.round(np.diff(config.azimuth_range)/config.azimuth_bin_size)+1)
    )
    df_azimuth = get_azimuth_hist(df, azimuth_bins, config.max_azimuth_distance, 
                                  to_capture_time=True)
    
    print("✓ Distributions computed")
    
    # ==================== Generate Plots ====================
    if not args.no_plots:
        print("\nGenerating plots...")
        try:
            plot_hunt(df, to_capture_time=True, video_path=video_path, save_fig=True)
            plot_approaches(df, to_capture_time=True, video_path=video_path, save_fig=True)
            plot_speeds_and_distance(df, mouse=True, cricket=True, to_capture_time=True, 
                                   contact_distance=config.contact_distance, 
                                   video_path=video_path, save_fig=True)
            plot_azimuth_hist(df, n_bins=20, approach_only=True, 
                            video_path=video_path, save_fig=True)
            print("✓ Plots saved")
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")
    
    # ==================== Generate Annotated Video ====================
    if not args.no_video:
        print("\nGenerating annotated video...")
        try:
            annotate_video(df, video_path, fps_out=10, to_capture_time=True, 
                         show_time=True, show_borders=True, label_bodyparts=True, 
                         show_approaches=True, show_approach_number=True, 
                         show_contacts=True, show_contact_number=True, 
                         show_azimuth=True, show_azimuth_lines=True, 
                         save_video=True, show_video=False, border_size=40, 
                         show_distance=True, show_speed=True, 
                         video_path_ext='_annotated_slow')
            print("✓ Annotated video saved")
        except Exception as e:
            print(f"Warning: Video generation failed: {e}")
    
    # ==================== Save Results ====================
    print("\nSaving results...")
    
    trial_id = video_path.split('/')[-1].split('.')[-2]
    summary_df = summarize_df(df, trial_id=trial_id, condition=args.condition)
    
    try:
        save_path = get_save_path_csv(h5_file)
        df.to_csv(save_path)
        print(f"✓ Main dataframe: {save_path}")
        
        save_path_summary = get_save_path_csv_summary(h5_file)
        summary_df.to_csv(save_path_summary)
        print(f"✓ Summary: {save_path_summary}")
        
        save_path_distribution = get_save_path_csv_distribution(h5_file)
        df_distribution.to_csv(save_path_distribution)
        print(f"✓ Distribution: {save_path_distribution}")
        
        save_path_density = get_save_path_csv_density(h5_file)
        df_density.to_csv(save_path_density)
        print(f"✓ Density: {save_path_density}")
        
        save_path_azimuth = get_save_path_csv_azimuth(h5_file)
        df_azimuth.to_csv(save_path_azimuth)
        print(f"✓ Azimuth: {save_path_azimuth}")
        
    except Exception as e:
        print(f"Warning: Some outputs failed to save: {e}")
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
