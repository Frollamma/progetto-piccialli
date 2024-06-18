Dataset containing features extracted from images of wells containing gold nanoparticles that bind to bacteria in the liquid and change color depending on the detected concentration.

Columns:

- Labels:
  - ‘value_target’ (numeric): the target value (the gold concentration value)
  - 'class_target' (categorical): the target class (the concentration of bacteria)
- Well features:
  - 'well_plate_name' (categorical): well plate numbering
  - 'wp_image_version' (categorical): Indicates the version of the well plate image, as multiple photos are taken of the same well plate. Our own interpretation: it indicates all the camera settings together, changing version means changing camera settings and vice versa.
  - 'well_name' (categorical): name of a single well in the well plate. In a standard 96-well plate, the rows are generally labeled with the letters A-H and the columns are labeled with the numbers 1-12.
    Therefore, the well in the top left corner will be well A1, while the well in the bottom right corner will be well H12.
  - 'wp_image_prop' (numeric): no description given
- Camera features:
  - 'FNumber' (numeric): the aperture opening in the camera lens
  - 'ISOSpeedRatings' (numeric): the sensitivity of the camera sensor to light
  - 'Orientation' (numeric): the camera orientation
  - 'ExposureTime' (numeric): the length of time the camera shutter remains open to expose the camera sensor to light
  - 'ExposureBiasValue' (numeric): no description given
  - 'FocalLength' (numeric): the focal length of the camera lens
  - 'Flash' (numeric): indicates whether the flash was fired
  - 'MeteringMode' (numeric): the method used by the camera to measure light for exposure calculation
  - 'BrightnessValue' (numeric): the brightness of the image
  - 'ApertureValue' (numeric): the aperture setting of the camera lens
  - 'MaxApertureValue' (numeric): no description given
  - 'ShutterSpeedValue' (numeric): the shutter speed setting of the camera
- Channel features. CHANNEL is one of the following: 'full_gray', 'full_blue', 'full_red', 'full_green', 'full_L','full_a','full_b', 'full_H', 'full_S', 'full_V', 'gray', 'blue', 'green', 'red', 'L', 'a', 'b', 'H', 'S', 'V'
  - [CHANNEL]\_mean (numeric): the average intensity or brightness of the pixels within the specified color channel
  - [CHANNEL]\_mean\_[MEASURE] (numeric): the mean value of the respective measure
  - [CHANNEL]\_stddev (numeric): the standard deviation of pixel intensities within the specified color channel
  - [CHANNEL]\_mean_trm30 (numeric): the mean intensity of pixels after removing the top and bottom 30% of extreme values
  - [CHANNEL]\_mean_PIL (numeric): the mean intensity of pixels after 'PIL' (Python Imaging Library) processing
  - [CHANNEL]\_skewness (numeric): the asymmetry of the pixel intensity distribution within the specified color channel
  - [CHANNEL]\_entropy (numeric): the randomness or unpredictability of pixel intensities within the specified color channel
  - [CHANNEL]\_entropy2 (numeric): another calculation of entropy
  - [CHANNEL]\_entropy_glcm (numeric): Entropy calculated based on the Gray Level Co-occurrence Matrix (GLCM)
- Other features
  - 'mock' (categorical): no description given. We interpreted this as a way of saying if the record was generated or not. It has only the value of `False`
