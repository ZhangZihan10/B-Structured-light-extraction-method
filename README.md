# \# Structured-light-extraction-method

# 

# This repository provides a MATLAB-based implementation of our structured-light extraction programs. It focuses on the robust and high-precision extraction of laser stripe centerlines from complex fisheye images, serving as a critical foundational step for subsequent 3D reconstruction and calibration tasks.

# 

# \## 🔗 Project Resources

# 

# \* \*\*Video Demonstration:\*\* \[Watch the algorithm explanation and demo on Bilibili](https://www.bilibili.com/video/BV1dK9RBLEXK/?spm\_id\_from=333.1387.homepage.video\_card.click\&vd\_source=e0a6a568a12a54c6b8cab40567343784)

# 

# \## ✨ Core Features

# 

# \* \*\*Single-Target Adaptive Extraction:\*\* Handles complex optical interactions, such as extracting standard red lasers using row projection, isolating weak magenta lasers on blue surfaces via custom color scoring, and applying PCA (Principal Component Analysis) fitting for inclined laser lines on dark targets.

# \* \*\*Multi-Target Synchronous Extraction:\*\* Integrates a trained \*\*DeepLabV3+ (ResNet-50)\*\* semantic segmentation model. It automatically identifies multiple objects in a single frame, using these semantic masks to filter out global background interference (e.g., robot arms, table reflections).

# \* \*\*Robust Noise Filtering:\*\* Utilizes HSV/RGB color space conversions, area filtering, and morphological operations to generate clean, continuous centerlines.

# 

# \## ⚙️ Environment \& Requirements

# 

# \* \*\*Platform:\*\* MATLAB

# \* \*\*Dependencies:\*\* \* Image Processing Toolbox

# &#x20; \* Deep Learning Toolbox (required for loading the `resnet50-50.mat` semantic segmentation model)

# 

# \## 🚀 System Logic Flow

# 

# 1\. \*\*Single-Target Modules:\*\* Used to verify individual extraction logic based on specific surface properties (e.g., fixed ROI, enhancing target-specific colors).

# 2\. \*\*Overall Program:\*\* \* Input fisheye image ➔ Semantic Segmentation (DeepLabV3+) ➔ Isolate target ROIs ➔ Global candidate extraction ➔ Intersect candidates with target masks ➔ Output final centerlines and class assignments.

# 

# \## 📝 Acknowledgements

# 

# \* \*(To be updated)\*

