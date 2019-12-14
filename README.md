## Understanding AI Video Surveillance: What Makes Anomaly Detection So Difficult?
### Elena Ramlow

This project uses manually annotated surveillance video to analyze and compare the predictive abilities of two object detection models--Faster RCNN and SSD. Pretrained models are obtained from Tensorflow Detection Zoo and transfer learning is applied. The files used in data cleaning and processing and model training and analysis are the following:

### 'data_processing.ipynb'

This file builds off of code from a <a href="https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d">tutorial<a/> on Medium for object detection in video using Tensorflow models. First it splits the video data into training and testing sets, then resizes the videos to 240 by 320 pixels and sets the frames per second at 30. It then creates tf records for training, validation, and testing data to be used in retraining tensorflow models using 'create_tf.py' and 'create_testtf.py'. It then outputs train.record, val.record, and test.record, which are the tf records, and trainval.txt and testval.txt, which contain lists of each frame grouped in training and testing.

### 'create_tf.py' and 'create_testtf.py'

These files use code adapted from Tensorflow's <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py"><create_pascal_tf_record.py<a/> and a <a href = "https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d">tutorial<a/> published on Medium. The first creates tf records for training and validation data and the second does the same for testing data.

### 'file_record.csv'

This csv file contains a list of each Abuse video file name the number of frames, and whether it should be used in training or testing. The original file used in cleaning the data contained information for all files in UCF-Crime, however, it has been amended to only the Abuse videos and random assignment is used to create 25/75 split of testing/training data. The low proportion of training videos is intended to simulate the real-word, where anomolies are unique in nature and difficult to identify from one situation to another. 

### 'FasterRCNN.ipynb'

This file is run in Google Colab and the code is adapted from another <a href = "https://medium.com/analytics-vidhya/training-an-object-detection-model-with-tensorflow-api-using-google-colab-4f9a688d5e8b">Medium tutorial<a/>. It initiates tensorflow and Python, sets everything up, downloads the pretrained Faster RCNN model from Tensorflow Detection Zoo and retrains the final layers on the new data provided in tf.record format. The model is then saved and evaluated using the training data, validation data, and testing data. An example test image is fed into the model to produce a visualized predicted bounding box and class prediction. A label_map.pbtxt file was created to specify the Abuse class for use in this workbook. This file also uses the faster_rcnn.config, faster_rcnn2.config, and faster_rcnn3.config files which were editted from the configuration file from Tensorflow to this task's specifications.

### 'SSD.ipynb'

This file is run in Google Colab and the code is adapted from another <a href = "https://medium.com/analytics-vidhya/training-an-object-detection-model-with-tensorflow-api-using-google-colab-4f9a688d5e8b">Medium tutorial<a/>. It initiates tensorflow and Python, sets everything up, downloads the pretrained SSD model from Tensorflow Detection Zoo and retrains the final layers on the new data provided in tf.record format. The model is then saved and evaluated using the training data, validation data, and testing data. An example test image is fed into the model to produce a visualized predicted bounding box and class prediction. A label_map.pbtxt file was created to specify the Abuse class for use in this workbook. This file also uses the SSD.config, SSD2.config, and SSD3.config files which were editted from the configuration file from Tensorflow to this task's specifications.

Also contained in this repository are a formal writeup of the project following conference paper format and a poster used in presentation of the project. 

##### A full repository included the video data, cleaned data, video frames, as well as the imported files from Tensorflow can be found on <a href = "https://drive.google.com/drive/u/1/folders/1EXZdWIvNb9m16urtD-F5DCh9Yoh-4HMX">Google Drive<a/>. 

