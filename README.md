# BC CSCI-3343: Human Counter

In our project we will be creating a multi-use live human counter.  It will allow users to utilize different object detection algorithms, as well as different input types, such as videos and images.  


## Installation



### Quick Install using [pip](https://pip.pypa.io/en/stable/):
```bash
pip install opencv-python
pip install numpy
pip install imutils
pip install argparse
```



## Usage
Navigate to the root directory

```python
python main.py -i "SomePath/ToAnImage/Here.jpg" -od 'HOG'
python main.py -v "SomePath/ToVideo/Here.mp4" -od 'HOG'

python main.py -i "SomePath/ToAnImage/Here.jpg" -od 'YOLO'
```




## Different Object Detection Methods

### YOLO (Matt)
YOLO - standing for You Only Look Once, is a more efficient and faster algorithm used for object detection. Unlike HOG, which repurposes classifiers to detect things, YOLO makes use of an end-to-end neural network that predicts bounding boxes and class all at the same time. It essentially divides an image into N number of S x S grids. Each cell of the grid predicts bounding boxes and confidence score. This method does require more computational power, and has issues with detecting smaller objects compared to other methods.

### Faster R-CNN (Sean)
This is a description

### HOG w CVM (AJ)
There are seven stages to this model and implementation of the algorithm. It begins with extraction of the HOG features from the data set using samples. You train the positive and negative samples to create the model. These trained models are then used to generate detectors, which are used to identify the ‘false positive’ tests, or where the model detected a human that was not actually there. You again extract the HOG features from these samples, further training the model. THe object is then identified and the detection area, or bounding box, is optimized. 





## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

