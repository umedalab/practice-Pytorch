Original code:
 - main.py with data from data_anime4classes
 expected accuracy 100% and show epochs and loss

Current code 4 classes:
 - python train_imageclassification_model.py
 expected not show any information during the training because the dataset is too small (it should be minimum 20 images for class).
 output: traced_model.pt (use for the C++ version)

Test C++:
 - cmake on cmakelist (this folder)
 - example-app.cpp (example)
 - Run. The program captures from the camera but override the content with images.
 

The virtual environment is located at:
envs\gLearning