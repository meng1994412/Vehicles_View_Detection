# Vehicles View Detection
## Objectives
Implemented two models for detecting the presence of traffic signs and differentiating the front and rear views of the vehicles in images or video streams.
* Built image datasets and image annotations in TensorFlow record format.
* Trained Faster R-CNN on the LISA Traffic Signs dataset to detect and recognize 47 United States traffic sign types.
* Trained SSD on the Davis King’s vehicles dataset to differentiate the front and rear views of the vehicles.
* Evaluate the accuracy and apply the trained Faster R-CNN and SSD models to input images and video streams.

**This is the second part of the project, which mainly focus on training Single Shot Detector model on differentiating the frontal and rear view of the vehicles. For the first part, please refer to [Traffic Sign Detection](https://github.com/meng1994412/Traffic_Sign_Detection) repo**

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.1.0
* [keras](https://keras.io/) 2.2.4
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
  * [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [scikit-learn](https://scikit-learn.org/stable/) 0.20.1
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/) 1.16.2
* [PIL](https://pillow.readthedocs.io/en/stable/) 5.4.1

## Approaches
The dataset is from [Davis King’s dlib library](http://dlib.net/files/data/) and was annotated by King for usage in a demonstration of his max-margin object detection algorithm. Each image in the dataset is captured from a camera mounted to a car’s dashboard. For each image, all visible front and rear views of vehicles are labeled as such.

### Setup configurations
In order properly import functions inside TensorFlow Object Detection API, `setup.sh`([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/setup.sh)) helps to update the `PYTHONPATH` variable. Thus, remember to source it every time, as shown below.
```
source setup.sh
```

The `dlib_front_rear_config.py` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/config/dlib_front_rear_config.py)) under `config/` directory contains the configurations for the project, including path to the annotations file and training & testing sets, training/testing split ratio, and classification labels.

### Build Tensorflow record dataset
According to Tensorflow Object Detection API, we need to have a number of attributes to makes up data points for object detection. They are including: (1) the Tensorflow encoded image, (2) the height and width of the image, (3) the file encoding of the image, (4) the filename of the image, (5) a list of bounding box coordinates (normalized in range [0, 1], for the image), (6) a list of class labels for each bounding box, (7) a flag used to encode if the bounding box is "difficult" or not.

The `tfannotation.py` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/pipeline/utils/tfannotation.py)) under `pipeline/utils/` directory build a class to encapsulate encoding an object detection data point in Tensorflow `record` format.

The `build_vehicle_records.py` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/build_vehicle_records.py)) convert raw image dataset and annotation file into Tensorflow `record` format dataset with corresponding class label file, in order to train a network by using Tensorflow Object Detection API.

We could use following command to build the Tensorflow `record` datasets.
```
time python build_vehicle_records.py
```

### Train & evaluate SSD model
In order to train the SSD model via transfer learning, we could download the pre-trained SSD model so we can fine-tune the network. And we also need to set up the Tensorflow Object Detection API configuration file for training.

All the pre-trained models can found in the [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). In this project, I use `ssd_inception_v2_coco` model, which is the SSD with GoogleNet(Inception V2) as base model pre-trained on COCO dataset.

The `ssd_vehicles.config` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/dlib_front_and_rear_vehicles_v1/experiments/training/ssd_vehicles.config)) under `dlib_front_and_rear_vehicles_v1/experiments/training/` directory sets up the Tensorflow Object Detection API configuration file for training.

The `model_main.py` ([check here](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py)) inside Tensorflow Object Detection API is used to train the model. The following command demonstrates the proper start of the training process (make sure we are current under `models/research/` directory of Tensorflow Object Detection API and source the `setup.sh` file).
```
python object_detection/model_main.py \
--pipeline_config_path=PATH_TO_CONFIGURATION \
--model_dir=PATH_TO_PRE_TRAINED_MODEL \
--num_train_steps=100000 \
--sample_1_of_n_eval_examples=1 \
--alsologtostderr
```

The `export_inference_graph.py` ([check here](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py)) in Tensorflow Object Detection API is used to export our model for inference. The following command can be used to export the model.
```
python object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path PATH_TO_CONFIGURATION
--trained_checkpoint_prefix PATH_TO_TRAINING_MODEL/model.ckpt-100000 \
--output PATH_TO_EXPORT_MODEL
```

### Apply SSD model to images and videos
The `predict_image.py` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/predict_image.py)) applys the network we trained to an input image outside the dataset it is trained on. And the `predict_video.py` ([check here](https://github.com/meng1994412/Vehicles_View_Detection/blob/master/predict_video.py)) apply to an input video.

The following command can apply SSD model to inference of images and videos.
```
python predict_image.py --model PATH_TO_EXPORT_MODEL/fronzen_inference_graph.pb --labels PATH_TO_CLASSES_FILE/classes.pbtxt --image SAMPLE_IMAGE.jpg --num_classes NUM_OF_CLASSES
```
```
python predict_video.py --model PATH_TO_EXPORT_MODEL/fronzen_inference_graph.pb --labels PATH_TO_CLASSES_FILE/classes.pbtxt --input SAMPLE_VIDEO.mp4 --output OUTPUT_VIDEO.mp4 --num_classes NUM_OF_CLASSES
```

## Results
### Evaluation of the SSD model
Figure 1 and Figure 2 below show the evaluation of the model including precision and recall of detection boxes. As we can see, the SSD model achieves 65.71% mAP @ 0.5 IoU.

<img src="https://github.com/meng1994412/Vehicles_View_Detection/blob/master/output/DetectionBoxes_Precision.png" height="400">

Figure 1: Precision evaluation of the model.

<img src="https://github.com/meng1994412/Vehicles_View_Detection/blob/master/output/DetectionBoxes_Recall.png" height="400">

Figure 2: Recall evaluation of the model.

### Apply SSD model to images and videos
Figure 3, Figure 4, and Figure 5 show some samples for detecting either frontal or rear view of the vehicles. However, there are some missing detection cases and false-positive cases.

<img src="https://github.com/meng1994412/Vehicles_View_Detection/blob/master/output/sample_1.png" height="300">

Figure 3: Sample #1 for detecting rear view of the vehicles on the highway.

<img src="https://github.com/meng1994412/Vehicles_View_Detection/blob/master/output/sample_2.png" height="500">

Figure 2: Sample #2 for detecting both frontal and rear views of the vehicles (missing detection situation exists).

<img src="https://github.com/meng1994412/Vehicles_View_Detection/blob/master/output/sample_3.png" height="300">

Figure 3: Sample #3 for detecting both frontal and rear views of the vehicles (missing detection situation exists).

## Next Step
The next step of the project is to find some techniques to increase the detection accuracy, and solve the missing detection cases and false positive cases.

## Traffic Sign Detection
For the second part of the project, please refer to [Traffic Sign Detection](https://github.com/meng1994412/Traffic_Sign_Detection) repo for more details.
