# -*- coding: utf-8 -*-
"""
Configuration parameters for hunting behavior analysis.

This module provides a centralized configuration class for all analysis parameters,
making it easy to adjust settings without modifying the main analysis code.

Author: KerschensteinerLab
"""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class AnalysisConfig:
    """Configuration parameters for hunting behavior analysis.
    
    All parameters are documented with their units and typical values.
    Modify these values to adjust analysis sensitivity and behavior detection.
    """
    
    # ==================== Video Parameters ====================
    frame_rate: float = 30.0
    """Video frame rate in frames per second (fps)"""
    
    # ==================== Arena Dimensions ====================
    arena_width: float = 45.0
    """Arena width in centimeters (x-axis)"""
    
    arena_height: float = 38.0
    """Arena height in centimeters (y-axis)"""
    
    # ==================== Detection Thresholds ====================
    speed_threshold: float = 10.0
    """Minimum mouse speed for approach detection (cm/s)"""
    
    contact_distance: float = 4.0
    """Maximum mouse-cricket distance for contact detection (cm)"""
    
    # ==================== Smoothing Parameters ====================
    window_size: int = 8
    """Window size for approach smoothing (frames)"""
    
    smooth_frames: int = 15
    """Number of frames for speed/acceleration Savitzky-Golay smoothing"""
    
    smooth_order: int = 3
    """Polynomial order for Savitzky-Golay filter"""
    
    acceleration_smooth_frames: int = 29
    """Number of frames for acceleration smoothing (should be odd)"""
    
    # ==================== Behavioral Criteria ====================
    body_azimuth_threshold: float = 60.0
    """Maximum absolute body angle relative to cricket for approach (degrees)"""
    
    diff_frames: int = 4
    """Number of frames to look back for speed change detection"""
    
    diff_speed: float = -20.0
    """Speed change threshold for deceleration detection (cm/s, negative)"""
    
    # ==================== Spatial Analysis ====================
    max_border_distance: float = 19.0
    """Maximum distance from border for spatial analysis inclusion (cm)"""
    
    bin_size: float = 1.0
    """Bin size for density histograms (cm)"""
    
    bin_num: int = 20
    """Number of bins for 2D density maps"""
    
    azimuth_bin_size: float = 5.0
    """Bin size for azimuth distributions (degrees)"""
    
    azimuth_range: Tuple[float, float] = (-180.0, 180.0)
    """Range for azimuth analysis (degrees)"""
    
    max_azimuth_distance: float = 5.0
    """Maximum mouse-cricket distance for azimuth analysis, per Hoy 2019 (cm)"""
    
    # ==================== Likelihood Thresholds ====================
    likelihood_cutoff: float = 0.9
    """Minimum DeepLabCut likelihood for accepting pose predictions (0-1)"""
    
    def get_bin_range(self) -> List[List[float]]:
        """Get the bin range for 2D density maps.
        
        Returns
        -------
        list of list of float
            [[y_min, y_max], [x_min, x_max]] for density binning
        """
        return [[0, self.arena_height], [0, self.arena_width]]
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises
        ------
        ValueError
            If any parameters are invalid or inconsistent
        """
        if self.frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        
        if self.arena_width <= 0 or self.arena_height <= 0:
            raise ValueError("Arena dimensions must be positive")
        
        if self.speed_threshold < 0:
            raise ValueError("speed_threshold must be non-negative")
        
        if self.contact_distance <= 0:
            raise ValueError("contact_distance must be positive")
        
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        
        if self.smooth_frames < 3 or self.smooth_frames % 2 == 0:
            raise ValueError("smooth_frames must be odd and at least 3")
        
        if self.acceleration_smooth_frames < 3 or self.acceleration_smooth_frames % 2 == 0:
            raise ValueError("acceleration_smooth_frames must be odd and at least 3")
        
        if self.smooth_order < 1 or self.smooth_order >= self.smooth_frames:
            raise ValueError("smooth_order must be between 1 and smooth_frames-1")
        
        if not (0 <= self.likelihood_cutoff <= 1):
            raise ValueError("likelihood_cutoff must be between 0 and 1")
    
    def __str__(self) -> str:
        """Return a formatted string representation of the configuration."""
        lines = ["Analysis Configuration:"]
        lines.append(f"  Frame rate: {self.frame_rate} fps")
        lines.append(f"  Arena: {self.arena_width} × {self.arena_height} cm")
        lines.append(f"  Speed threshold: {self.speed_threshold} cm/s")
        lines.append(f"  Contact distance: {self.contact_distance} cm")
        lines.append(f"  Window size: {self.window_size} frames")
        return "\n".join(lines)


# Default configuration instance
default_config = AnalysisConfig()


if __name__ == "__main__":
    # Test configuration
    config = AnalysisConfig()
    print(config)
    print("\nValidating configuration...")
    try:
        config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
