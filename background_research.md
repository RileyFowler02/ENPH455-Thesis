# Deep learning-based segmentation and quantitative analysis of lumbar paraspinal muscles and fat infiltration in multicenter T2-weighted MRI data
(Wang, et al. 2025)
## Background/Abstract
- Developed deep learning algorithms based on T2WI to accurately segment the LPS using 1457 participants.
- Compared 2D and 3D segmentation models
  - Used **Dice** and **IoU** as the main evaluation models:
    - *Dice* - Sorensen-Dice coefficient (Used to measure the overlap or similarity between two sets of data, primarily in medical image segmentation and computer vision): $\text{Dice}(A,B)=\frac{2*|A \cap B|}{|A|+|B|}$
    - *IoU* - Intersection over Union or Jaccard index measures how much a model's predicted region overlaps with the true ground truth-region (i.e., via a bounding box or segmentation mask): $\text{IoU}=\frac{\text{Area of Intersection}}{\text{Area of Union}}$
- Intramuscular fat was segmented using the maximum inter-class variance method (OTSU);
  - Automatic image thresholding and sementation technique
    - Goal: Separates an image's pixels into two distinct classes (typically foreground and background) by finding a single optimal intensity treshold ($T$).
    - Logic: Tests all possible intensity threshold values and selects the one that maximizes the variance between the two classes (hence inter-class)
  - Consistency between OTSU-based fat infiltration results and radiologist's was evaluated using the weighted Kappa test:
    - A statistical measure used to evaluate inter-rater reliability (agreement between two raters or observers) when the data consists of ordered categorical variables
## Methodology
1. MR Scan Parameters and Preprocessing
- Parameters:
  - Seven MR devices with three different field strengths from four manufacturers were used to acquire the training and test sets.
- Preprocessing:
  - Removed image intensities above 99.5% and below 0.05%
  - All images were resampled to isotropic voxel size (0.5 x 0.5 x 5.00 $\text{mm}^3$) using B-spline interpolation
  - Intensities were further normalized using a bin width fixed to 25
  - Finally, images were normalized by Z-score (how many standard deviations a data point is above or below the mean of a dataset) to obtain a standard normal distribution of image intensities
2. LPS Segmentation Algorithms Training and Testing
- An nnU-Net was trained to automatically segment the multifidus, erector spinae, and psoas major muscles
- The automatic internal procedures provided by the nnU-Net, such as data fingerprinting for a data specific network setup, pre-processing techniques were applied
- The 2D and 3D networks were trained together for 1000 epochs in all MRI orientations
- During the training process, the label-wise accuracy of the automatic segmentation of the 2D and 3D networks, as well as the ensemble result of these networks, were calculated as the mean Dice similarity coefficient (DSC), Jaccard coefficient (JC), hausdorff distance (HD), average surface distance (ASD) and average volume difference (AVD) relative to the manual ground truth
- Both DSC and JC are employed as pivotal metrics for evaluating the degree of similarity between predicted and actual outcomes, with their values constrained to a range between 0 and 1
- HD is a distinctive metric designed to quantify the disparity in similarity between the two point sets, range from 0 to positive infinity, with lower values indicating a higher degree of similarity between the two point sets
- ASD quantifies the degree of similarity by calculating the mean distance between all corresponding points on the two surfaces
  - A smaller value indicates that the two surfaces are more closely aligned geometrically, thereby suggesting a greater degree of similarity
- AVD is employed as a measure of the discrepancy between the two volumes
  - A smaller value indicates a smaller difference between the two volumes
- The Pytorch version 2.0.1, cuda version 11.7, and four 24 G Nvidia TITAN RTX graphics cards were used to train the MF, ES, and PS segmentation networks in this study
  - **THIS MIGHT BE MY ANGLE. CAN I DO THIS WITHOUT SUCH HEAVY COMPUTE?**
3. Fat segmentation and fat infiltration calculation
- 