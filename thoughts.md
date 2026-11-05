**GOAL:** *BUILD AN AUTOMATED LUMBAR MULTIFIDUS ASSESSMENT TOOL*

**Input:** Patient MRI data (Axial T2 DICOM series, potentially supported by sagittal T2 series)

**Step 1:** Image processing and anatomical localization (Organize scans, identify lumbar levels, select relevant slices)
 
**Step 2:** Multifidus segmentation (Identify and outline the left and right multifidus muscles)

**Step 3:** Quantitative measurements (Calculate cross-sectional area, asymmetry and potentially fat-related metrics)

**Output:** Quantitative assessment (Measurements, segmentation overlays, visualizations and a report)

---

### What do I have?
- Multiple patients:
  - each with multiple MRI series
  - each series containing multiple DICOM images that show slices of the muscle
- Axial T2 MRIs:
  - Horizontal cross-section through the body
  - Used to identify and segment the left and right multifidus muscles
- Sagittal T2 MRIs:
  - Viewed from the side
  - Helpful or understanding anatomy, identifying vertebral level and locate relevate axial slices

---

### Gates
#### Gate 1 - Understand and organize the dataset to know what I have and how it is structured
- Setting up 3D Slicer workspace
- Understanding pixel spacing, slice spacing, and image orientation
- Preprocessing
- **Gate passes when:** I have the data navigable and easily identifiable

### Gate 2 - Defining measurements to develop a protocol
- Which lumbar levels?
  - L3/L4
  - L4/L5
  - L5/S1
- How to seperate between left and right?
- One axial slice? Or entire volume?
- How to determine the boundaries?
- Measurements:
  - Left/right multifidus CSA (cross-sectional area)
  - Left-right asymmetry
  - Muscle shape/dimensions
  - Fat-related intensity metric
- **Gate passes when:** I have a written plan of how and what I will do measurement-wise

### Gate 3 - Manually segment reference dataset to create reference data
- For a small sample of patients:
  1. Select an axial slice at a known lumbar level.
  2. Open Segment Editor.
  3. Create separate segments for left and right multifidus.
  4. Manually trace the boundaries.
  5. Save the segmentation.
  6. Repeat across additional patients and levels.


### Gate 4 - Build a basic quantitiative measurement pipeline to convert manual segmentations into actual measurements
- With Python, I should be able to:
  1. Load the original DICOM.
  2. Load the corresponding segmentation mask.
  3. Count the pixels inside each muscle.
  4. Convert pixel area to mm² using DICOM spacing.
  5. Calculate left/right CSA and asymmetry.
  6. Save the measurements in a CSV file.
- **Gate passes when:** I can calculate physical CSA from a manual segmentation without requiring by-hand calculation

### Gate 5 - Automate the segmentation that can identify the multifidus without manual tracing
- Compare approaches:
  - Classical image processing/thresholding
  - 2D U-Net
  - Segment Anything
- Evaluate via:
  - Dice coef
  - CSA measurement error
  - Visual comparison

**Gate passes when:** You have a working automated segmentation model and have quantified how closely its measurements agree with your reference annotations.

### Gate 6 - Extending the measurements and visualizations into a full-fledged tool
Once segmentation works, consider:
- Measurements across multiple lumbar levels.
- Left/right muscle asymmetry.
- Fat-related intensity measurements.
- 3D visualization of the multifidus.
- A standardized patient-level report.

This is also where sagittal images may become useful for anatomical localization and organizing measurements across levels.

**Gate passes when:** You can produce a meaningful, repeatable set of measurements and visualizations from a complete study.

### Validate the system and demonstrate its use
Technical validation:
- Does the segmentation agree with manual annotations?
- Are CSA measurements accurate?
- Does it work across different patients and image quality?
- How long does it take compared with manual analysis?
Clinical/exploratory validation:
- Do measurements vary across lumbar levels?
- Are CSA measurements associated with RSNA stenosis severity grades?
- Are there patterns in left/right asymmetry?