# MangaCleaner 2.0
This project is my second attempt at tackling the problem of automatic detection of text regions in Manga and "cleaning" them. 


## Code
To run this code, you'll need a Python 3.6 environment with NumPy, PyTorch 17, and Opencv-contrib packages.

Command line args are not currently set up, so if you would like to see a pass through of the current code, you must change the variable "image" to reflect path of your test image, then run `main.py`.

If you would like to retrain the Convnet, you must run `convnet.py`

## Strategy
The approach which I took was to find likely regions of text (contained in `main.py`) then getting rid of false-positives using a convolutional neural network (detailed in `convnet.py`). An additional attempt at removing false positives was made with training an support vector machine (which seemed to be too strict)


