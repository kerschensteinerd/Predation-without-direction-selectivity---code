# Detailed Setup Instructions

This document provides comprehensive setup instructions for the analysis code supporting Krizan et al., PNAS, 2024.

## Table of Contents

1. [Python Environment Setup](#python-environment-setup)
2. [MATLAB Environment Setup](#matlab-environment-setup)
3. [External Dependencies](#external-dependencies)
4. [Data Requirements](#data-requirements)
5. [Verification](#verification)

---

## Python Environment Setup

### System Requirements

- **Operating System:** Windows, macOS, or Linux
- **Python Version:** 3.8, 3.9, 3.10, or 3.11
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk Space:** 500 MB for dependencies + space for data

### Installation Steps

#### Option 1: Using pip (Recommended)

```bash
# Navigate to repository
cd Krizan2024_predation-wo-direction-selectivity

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment
conda create -n predation python=3.9

# Activate environment
conda activate predation

# Install dependencies
pip install -r requirements.txt
```

### Dependency Details

The following packages are required:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Array operations, numerical computing |
| pandas | ≥1.3.0 | Data manipulation, CSV/Excel I/O |
| scipy | ≥1.7.0 | Signal processing (smoothing, filtering) |
| matplotlib | ≥3.5.0 | Plotting, visualization |
| opencv-python | ≥4.5.0 | Video processing, image manipulation |
| PyQt5 | ≥5.15.0 | GUI dialogs, interactive elements |
| tables | ≥3.7.0 | HDF5 file support (PyTables) |
| h5py | ≥3.6.0 | HDF5 file I/O |

### Troubleshooting Python Installation

**Issue:** `pip install` fails for PyQt5 on macOS
```bash
# Solution: Use conda
conda install pyqt
```

**Issue:** OpenCV import error
```bash
# Verify installation
python -c "import cv2; print(cv2.__version__)"

# Reinstall if needed
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

**Issue:** HDF5 library errors with `tables`
```bash
# On Ubuntu/Debian:
sudo apt-get install libhdf5-dev

# On macOS:
brew install hdf5

# Reinstall tables
pip install --no-cache-dir tables
```

---

## MATLAB Environment Setup

### System Requirements

- **MATLAB Version:** R2020b or later (tested up to R2023b)
- **Operating System:** Windows, macOS, or Linux
- **RAM:** 16 GB minimum, 32 GB recommended for large datasets
- **Disk Space:** 2 GB for toolboxes + space for data

### Required Toolboxes

Install the following MATLAB toolboxes (via MATLAB Add-On Explorer):

1. **Signal Processing Toolbox** - For filtering, spectral analysis
2. **Image Processing Toolbox** - For receptive field mapping
3. **Statistics and Machine Learning Toolbox** - For curve fitting, statistical tests
4. **Optimization Toolbox** - For dose-response curve fitting (PLR analysis)

### Verifying Toolbox Installation

In MATLAB, run:

```matlab
% Check installed toolboxes
ver

% Test specific toolboxes
license('test', 'Signal_Toolbox')
license('test', 'Image_Toolbox')
license('test', 'Statistics_Toolbox')
license('test', 'Optimization_Toolbox')
```

### Setting Up MATLAB Path

```matlab
% Add repository directories to MATLAB path
addpath(genpath('/path/to/Krizan2024_predation-wo-direction-selectivity'))

% Save path for future sessions
savepath
```

---

## External Dependencies

### DeepLabCut (for Pose Tracking)

**Purpose:** Generate `.h5` pose tracking files from videos

**Installation:**
```bash
# Create separate environment
conda create -n deeplabcut python=3.8
conda activate deeplabcut

# Install DeepLabCut
pip install deeplabcut[gui]
```

**Usage:** See [DeepLabCut documentation](https://github.com/DeepLabCut/DeepLabCut)

**Note:** DeepLabCut is only needed if generating new tracking data. Pre-analyzed `.h5` files can be used directly.

---

### Kilosort3 (for Spike Sorting)

**Purpose:** Process raw electrophysiology recordings into spike times

**Installation:**
```bash
# Clone repository
git clone https://github.com/MouseLand/Kilosort.git
cd Kilosort

# Follow MATLAB GPU setup in Kilosort documentation
```

**Requirements:**
- MATLAB 2020b or later
- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit 11.0 or later

**Note:** Kilosort3 is only needed if processing raw neural recordings. Pre-sorted spike data can be used directly.

---

### npy-matlab (for Loading NumPy Files in MATLAB)

**Purpose:** Load Kilosort3 output files (`.npy` format) in MATLAB

**Installation:**

```bash
# Clone repository
git clone https://github.com/kwikteam/npy-matlab.git
```

**Setup in MATLAB:**

```matlab
% Add to MATLAB path
addpath('/path/to/npy-matlab')
savepath

% Test
test_data = readNPY('spike_times.npy');
```

---

## Data Requirements

### Behavioral Analysis Input Files

1. **Video File** (`.mp4`)
   - Top-down view of arena
   - Mouse and cricket visible
   - Recommended: 30 fps, 640×480 pixels minimum
   - Example: `trial_001.mp4`

2. **Pose Tracking File** (`.h5`)
   - DeepLabCut output
   - Required body parts: `nose`, `l_ear`, `r_ear`, `tail_base`, `cricket`
   - Example: `trial_001DLC_resnet50_cricket_huntingJan1shuffle1_100000.h5`

3. **Analysis CSV** (`.csv`)
   - Preprocessed tracking coordinates
   - Columns: frame_number, time, body part x/y/likelihood
   - Example: `trial_001_Analysis.csv`

### Electrophysiology Input Files

1. **Kilosort3 Output:**
   - `spike_times.npy` - Spike timestamps
   - `spike_clusters.npy` - Cluster assignments
   - `spike_templates.npy` - Template assignments
   - `channel_positions.npy` - Probe geometry

2. **Stimulus Timing:**
   - Allegro `.XDat` files with analog signals
   - Includes photodiode, trigger channels

### Pupillometry Input Files

1. **ISCAN Eye Tracking Data:**
   - Excel files (`.xlsx` or `.xls`)
   - Columns: time, pupil_area, illuminance
   - Multiple trials per file

---

## Verification

### Testing Python Installation

Create a test script `test_installation.py`:

```python
#!/usr/bin/env python
"""Test Python environment setup."""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    packages = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'cv2',
        'PyQt5',
        'tables',
        'h5py'
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ All packages imported successfully!")
        sys.exit(0)

if __name__ == '__main__':
    test_imports()
```

Run the test:

```bash
python test_installation.py
```

### Testing MATLAB Installation

In MATLAB, run:

```matlab
% Test basic functionality
fprintf('Testing MATLAB setup...\n');

% Test toolboxes
toolboxes = {'Signal_Toolbox', 'Image_Toolbox', ...
             'Statistics_Toolbox', 'Optimization_Toolbox'};

for i = 1:length(toolboxes)
    if license('test', toolboxes{i})
        fprintf('✓ %s available\n', toolboxes{i});
    else
        fprintf('✗ %s NOT available\n', toolboxes{i});
    end
end

% Test npy-matlab if available
try
    which readNPY
    fprintf('✓ npy-matlab available\n');
catch
    fprintf('✗ npy-matlab NOT available\n');
end

fprintf('\nSetup verification complete!\n');
```

---

## Platform-Specific Notes

### Windows

- Use Command Prompt or PowerShell for bash commands
- Path separators: Use `\` or raw strings in Python
- Video codec: Ensure H.264 codec installed

### macOS

- May need Xcode Command Line Tools: `xcode-select --install`
- PyQt5 installation: Consider using conda
- Video rendering: Ensure FFmpeg installed via Homebrew

### Linux

- Install system dependencies first:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install python3-dev libhdf5-dev ffmpeg
  
  # Fedora/RHEL
  sudo dnf install python3-devel hdf5-devel ffmpeg
  ```

---

## Getting Help

If you encounter issues not covered here:

1. Check the main `README.md` Troubleshooting section
2. Verify all dependencies are correctly installed
3. Open an issue on GitHub with:
   - Error message
   - Operating system and versions
   - Steps to reproduce

---

## Quick Start Verification

Once setup is complete, verify the installation:

```bash
# Python
cd predation
python hunting_analysis_script.py --help

# MATLAB
matlab -nodisplay -r "cd in_vivo_ephys; which dsRfMbParser; exit"
```

If both commands succeed without errors, your environment is ready!
