# BC CSCI-3343: Human Counter

Public safety has become an important task for many local government and health officials who keep track of people’s activity. Understanding an area's max capacity and how the space is utilized is an important part in ensuring public safety standards are held to a high standard. We aimed to design and train a model that can observe a space and keep track of its capacity as well as how the space is utilized. In our project we will be creating a multi-use live human counter.  It will allow users to utilize different object detection algorithms, such as HOG w OpenCV, HOG w CVM, and YOLO. It accepts both video and image inputs. 

  


## Installation
Our project requires a few dependencies, such as opencv-python, numpy, imutils and argparse.  While imutils and argparse are mostly utility
functions which allow for an easier development experience, opencv-python and numpy are more essential to our programs.  Opencv-python is the 
main driving part of our code for both our YOLO and HOG methods, and allows us to use tested and proven implementations
of object detection for our human counter.  (Don't reinvent the wheel!)

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

### HOG w OpenCV
HOG is a ‘feature descriptor’, which takes an image and simplifies it by assigning ‘features’ to important information within the image, and gets rid of unnecessary information. The process begins by calculating the gradient, and from that getting the magnitude and direction. A Histogram of gradients is calculated in an 8x8 cell, and then normalized into a block of 16x16 cells. This all builds a Histogram of Oriented Gradients feature vector, concentrating more on the borders and the shape of the images to extract its features individually from the background. Overall, it's less efficient compared to other approaches such as Faster RCNN and takes longer to train.



### YOLO (Matt)
#### Description of Algorithm
YOLO - standing for You Only Look Once, is a more efficient and faster algorithm used for object detection. Unlike HOG, which repurposes classifiers to detect things, YOLO makes use of an end-to-end neural network that predicts bounding boxes and class all at the same time. It essentially divides an image into N number of S x S grids. Each cell of the grid predicts bounding boxes and confidence score. This method does require more computational power, and has issues with detecting smaller objects compared to other methods.

#### Work Done
https://pjreddie.com/darknet/yolo/

- Created the YOLO class, which will allow for better workability and abstractability in the future.  Used the cv2.dnn framework with the darknet
library.  
- Created main.py class and added the option to run different algorithms through the command line. (not fully implemented)
- Grabbed the Darknet Reference Model config and weights from the above darknet reference.




### HOG w CVM (AJ)
#### Description
There are seven stages to this model and implementation of the algorithm. It begins with extraction of the HOG features from the data set using samples. You train the positive and negative samples to create the model. These trained models are then used to generate detectors, which are used to identify the ‘false positive’ tests, or where the model detected a human that was not actually there. You again extract the HOG features from these samples, further training the model. THe object is then identified and the detection area, or bounding box, is optimized.

## Demo/Results
#### Work Done

[Videos/images here]





## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

