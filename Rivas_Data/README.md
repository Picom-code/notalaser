## Important Information Before Running

The baseline 3D CNN model has already been executed on the dataset, and the results have been saved for reference. The script `format_test_dataset.py` was used to convert the raw MATLAB files into a format compatible with the model. You are welcome to run this script on your own data. If executed within Box in its current directory, it should operate smoothly.

Predictions were generated using `test_dataset_summary.py`, which you can reuse to replicate the results. Output files are stored in the `Test_Dataset_Results` folder. Each prediction is saved as a PNG image, displayed side-by-side with the corresponding `x.npy` ground truth labels. Additionally, the console output has been saved to `test_results.txt`.

These results were produced using the final baseline 3D CNN model checkpoint, `big_epoch_4_finished.pth`, located in the `pred` folder within the main directory.

Feel free to rerun any of the provided scripts on your own data. We pre-generated the results to minimize potential issues during replication.
