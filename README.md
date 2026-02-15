# Predation without Direction Selectivity - Analysis Code

This repository contains the analysis code (Python and MATLAB) supporting the publication:

**Krizan et al., "Predation without direction selectivity", PNAS, 2024**

The code analyzes multi-modal neuroscience data investigating whether direction-selective neural circuits are necessary for visually-guided predatory hunting behavior in mice.

## Overview

The repository contains three main analysis pipelines:

1. **Behavioral Analysis** (`predation/`) - Python scripts for analyzing mouse hunting behavior from video pose tracking
2. **Electrophysiology Analysis** (`in_vivo_ephys/`) - MATLAB scripts for quantifying neural direction selectivity
3. **Pupillometry Analysis** (`pupillary_light_responses/`) - MATLAB scripts for analyzing pupillary light responses

## Installation

### Python Environment (for Behavioral Analysis)

**Requirements:**
- Python 3.8 or later
- pip package manager

**Setup:**

```bash
# Clone the repository
git clone https://github.com/kerschensteinerd/Krizan2024_predation-wo-direction-selectivity.git
cd Krizan2024_predation-wo-direction-selectivity

# Install Python dependencies
pip install -r requirements.txt
```

### MATLAB Environment (for Electrophysiology & Pupillometry)

**Requirements:**
- MATLAB R2020b or later
- Signal Processing Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

**Additional Dependencies:**
- [Kilosort3](https://github.com/MouseLand/Kilosort) - For spike sorting (electrophysiology analysis)
- [npy-matlab](https://github.com/kwikteam/npy-matlab) - For loading NumPy files in MATLAB

See `SETUP.md` for detailed installation instructions.

## Usage

### Behavioral Analysis (Python)

The main script for analyzing hunting behavior is `predation/hunting_analysis_script.py`.

**Input Files:**
- `.mp4` video file of hunting trial
- `.h5` pose tracking file from DeepLabCut
- `_Analysis.csv` file with preprocessed tracking data

**Command-line Usage:**

```bash
cd predation

# Interactive mode (GUI file selection)
python hunting_analysis_script.py

# Command-line mode (automated)
python hunting_analysis_script.py \
    --video /path/to/video.mp4 \
    --h5 /path/to/tracking.h5 \
    --csv /path/to/analysis.csv \
    --capture-frame 1500 \
    --output-dir ./results
```

**Parameters:**

You can customize analysis parameters by editing `config.py` or passing command-line arguments. Key parameters include:

- `--frame-rate`: Video frame rate (default: 30 fps)
- `--arena-width`: Arena width in cm (default: 45)
- `--arena-height`: Arena height in cm (default: 38)
- `--speed-threshold`: Minimum speed for approach detection (default: 10 cm/s)
- `--contact-distance`: Distance threshold for contact detection (default: 4 cm)

**Output:**

The script generates:
- Annotated videos showing detected behaviors
- Summary statistics in Excel format
- Distribution plots (spatial, temporal, angular)
- Trajectory visualizations

### Electrophysiology Analysis (MATLAB)

**Workflow:**

1. Parse spike data and stimulus timing:
   ```matlab
   % In MATLAB
   cd in_vivo_ephys
   dsRfMbParser  % Opens file dialog
   ```

2. Analyze direction selectivity and receptive fields:
   ```matlab
   dsRfMbAnalysis  % Loads parsed data, computes selectivity indices
   ```

3. Visualize results:
   ```matlab
   mbRfPlotRates  % Generate polar plots and RF maps
   ```

### Pupillometry Analysis (MATLAB)

**Workflow:**

1. Parse eye tracking data:
   ```matlab
   cd pupillary_light_responses
   ExcelfromTDA_parser  % Opens file dialog
   ```

2. Analyze individual experiment:
   ```matlab
   PLR_analysis_individual  % Fits dose-response curve
   ```

3. Batch process multiple animals:
   ```matlab
   PLR_analysis_multiple  % Processes all parsed files
   ```

## Input Data Format

### Behavioral Analysis

**Video Files (`.mp4`):**
- Top-down view of mouse in arena with cricket
- Recommended: 30 fps, 640x480 or higher resolution

**Pose Tracking (`.h5`):**
- DeepLabCut output in HDF5 format
- Required body parts: `nose`, `l_ear`, `r_ear`, `tail_base`, `cricket`
- Each body part has: `x`, `y`, `likelihood` columns

**Analysis CSV (`.csv`):**
- Preprocessed tracking data with columns:
  - Frame number, time, body part coordinates
  - Optional: pre-computed features

### Electrophysiology

**Input Files:**
- Kilosort3 output: `spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy`
- Allegro XDat files with analog signals (stimulus timing)

### Pupillometry

**Input Files:**
- ISCAN eye tracking data exported to Excel
- Columns: time, pupil area, illuminance levels

## Configuration

Analysis parameters can be customized in `predation/config.py`:

```python
from predation.config import AnalysisConfig

config = AnalysisConfig()
config.frame_rate = 25.0  # Adjust for your recording setup
config.speed_threshold = 15.0  # Change behavior detection sensitivity
```

## Repository Structure

```
.
├── predation/                      # Behavioral analysis (Python)
│   ├── hunting_analysis_script.py   # Main analysis script
│   ├── hunting_analysis_functions.py # Core function library
│   ├── config.py                     # Configuration parameters
│   └── cricket_adjust_labels.py      # Pose tracking refinement
│
├── in_vivo_ephys/                  # Electrophysiology analysis (MATLAB)
│   ├── dsRfMbParser.m                # Parse spike & stimulus data
│   ├── dsRfMbAnalysis.m              # Compute selectivity indices
│   └── mbRfPlotRates*.m              # Visualization
│
├── pupillary_light_responses/      # Pupillometry analysis (MATLAB)
│   ├── ExcelfromTDA_parser.m         # Parse eye tracking
│   ├── PLR_analysis_individual.m     # Dose-response fitting
│   └── PLR_analysis_multiple.m       # Batch processing
│
├── requirements.txt                # Python dependencies
├── SETUP.md                        # Detailed setup instructions
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Computational Resources

**Python Analysis:**
- RAM: 8 GB minimum, 16 GB recommended
- Disk: ~100 MB per analyzed trial (videos, plots, data)
- Processing time: ~5-10 minutes per trial

**MATLAB Analysis:**
- RAM: 16 GB minimum for large recordings
- Disk: ~1 GB per recording (spike data, RF maps)
- Processing time: ~10-30 minutes per recording

## Troubleshooting

### Python Issues

**Problem:** `ImportError: No module named 'cv2'`
- **Solution:** Install OpenCV: `pip install opencv-python`

**Problem:** `FileNotFoundError` when loading video
- **Solution:** Use absolute paths or verify working directory

**Problem:** Arena calibration fails
- **Solution:** Ensure video frame 0 shows clear arena corners

**Problem:** `tables` module not found when loading `.h5` files
- **Solution:** Install PyTables: `pip install tables`

### MATLAB Issues

**Problem:** `readNPY` function not found
- **Solution:** Install npy-matlab and add to MATLAB path

**Problem:** Kilosort data not loading
- **Solution:** Verify Kilosort3 output files are in expected directory

**Problem:** Hardcoded path error (`E:\Projects\...`)
- **Solution:** This has been fixed - scripts now use file dialog or relative paths

## Citation

If you use this code, please cite:

```bibtex
@article{krizan2024predation,
  title={Predation without direction selectivity},
  author={Krizan, [First names] and [Other authors]},
  journal={Proceedings of the National Academy of Sciences},
  year={2024},
  publisher={National Academy of Sciences}
}
```

## License

This code is released under the MIT License. See `LICENSE` file for details.

## Contact

For questions or issues:
- Open an issue on GitHub
- Contact: KerschensteinerLab

## Acknowledgments

- DeepLabCut for pose estimation
- Kilosort3 for spike sorting
- MATLAB & Python scientific computing communities
