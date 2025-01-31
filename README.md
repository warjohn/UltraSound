Segmentation and NRRD Conversion Library

This library allows you to create segmentation masks and NRRD images from original DICOM files.

### Examples 
    us = UltrasoundSegmentation(input_path="test")
    us.dicom2nrrdImage()

This code will save your converted image No. 1 in nrrd format in the directory where you are working

    us = UltrasoundSegmentation(input_path="test")
    us.us.dicom2nrrdMask()

This code will create a mask for image No. 2 and save it in the working directory

