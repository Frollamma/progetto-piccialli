Dataset containing features extracted from images of wells containing gold nanoparticles that bind to bacteria in the liquid and change color depending on the detected concentration.

Columns:

- Labels:
  - ‘value_target’: the target value (the concentration value)
  - 'class_target': the target class
- Well features:
  - 'well_plate_name': well plate numbering
  - 'wp_image_version': Indicates the version of the wellplate image, as multiple photos are taken of the same wellplate.
  - 'well_name': name of a single well in the well plate. In a standard 96-well plate, the rows are generally labeled with the letters A-H and the columns are labeled with the numbers 1-12.
    Therefore, the well in the top left corner will be well A1, while the well in the bottom right corner will be well H12.
- Camera features:
  - 'FNumber': the aperture opening in the camera lens
  - 'ISOSpeedRatings': the sensitivity of the camera sensor to light
  - 'Orientation': the camera orientation
  - 'ExposureTime': the length of time the camera shutter remains open to expose the camera sensor to light
  - 'FocalLength': the focal length of the camera lens
  - 'Flash': indicates whether the flash was fired
  - 'MeteringMode': the method used by the camera to measure light for exposure calculation
  - 'BrightnessValue': the brightness of the image
  - 'ApertureValue': the aperture setting of the camera lens
  - 'ShutterSpeedValue': the shutter speed setting of the camera
- Channel features. CHANNEL is one of the following: 'full_gray', 'full_blue', 'full_red', 'full_green', 'full_L','full_a','full_b', 'full_H', 'full_S', 'full_V', 'gray', 'blue', 'green', 'red', 'L', 'a', 'b', 'H', 'S', 'V'
  - [CHANNEL]\_mean: the average intensity or brightness of the pixels within the specified color channel
  - [CHANNEL]\_mean\_[MEASURE]: the mean value of the respective measure
  - [CHANNEL]\_stddev: the standard deviation of pixel intensities within the specified color channel
  - [CHANNEL]\_mean_trm30: the mean intensity of pixels after removing the top and bottom 30% of extreme values
  - [CHANNEL]\_mean_PIL: the mean intensity of pixels after 'PIL' (Python Imaging Library) processing
  - [CHANNEL]\_skewness: the asymmetry of the pixel intensity distribution within the specified color channel
  - [CHANNEL]\_entropy: the randomness or unpredictability of pixel intensities within the specified color channel
  - [CHANNEL]\_entropy2: another calculation of entropy
  - [CHANNEL]\_entropy_glcm: Entropy calculated based on the Gray Level Co-occurrence Matrix (GLCM)
