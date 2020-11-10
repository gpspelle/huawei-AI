# Optical network modelling
Modelling the input-output (regression) function of an optical cascade

# How to create and test your model locally
1. Clone the following repo: https://github.com/ramp-kits/optical_network_modelling and extract the data inside the folder `./data`
2. Create a folder under "submissions" with the name of your model and a "regressor.py" file.
3. Inside regressor.py, create a `Regressor()` model, with `__init__()`, `fit()` and `predict()` methods.
4. Run `ramp-test --submission <model_folder_name>` to get results.

# How to create and test your model on Colab
1. Access https://colab.research.google.com/drive/1T_qn8oubwkQvEbZkwBl8Y8phhnj3rvP2?usp=sharing
2. Upload and extract the data
3. Upload your model under submissions
4. Run the cells
