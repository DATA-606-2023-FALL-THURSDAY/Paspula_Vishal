### 1. Title and Author
Prepared for UMBC Data Science Master Degree Capstone by Dr Chaoji (Jay) Wang
Author: Vishal Yadav Paspula, UMBC, Fall-2023 Semester
LinkedIn Profile:  https://www.linkedin.com/feed/
GitHub Project Link: https://github.com/VishalYadavPaspula/UMBC-DATA606-FALL2023-THURSDAY/blob/main/Vishal_606_Captson_project.md 
Presentation Link: Paspula_Vishal_Capstone_Project.pptx

### Background

The project focuses on the detection and identification of deep fake images, leveraging a sophisticated blend of machine learning techniques. At its core, the project aims to differentiate between real and manipulated images, particularly those that have been altered using advanced deep learning methods. By analyzing a dataset comprising both authentic and altered images, the project endeavors to train a predictive model capable of accurately identifying deep fakes.

The significance of this project lies in its potential to address growing concerns around digital authenticity and misinformation. In an era where image manipulation technologies are becoming increasingly accessible and sophisticated, the ability to reliably identify deep fakes is crucial for maintaining the integrity of digital media, safeguarding personal privacy, and combating the spread of false information. This project, therefore, holds substantial value in enhancing digital security and trustworthiness.

### Data Cleaning and Preprocessing:
Load the Dataset: The dataset, sourced from Zenodo, contains manipulated (deep fake) and real images of human faces. Each image is 256x256 pixels in JPG format.
Data Inspection: The notebook includes code to inspect images for inconsistencies, such as varied sizes or formats. All images are confirmed to be 256x256 JPGs.
Handling Missing/Corrupted Images: The notebook contains a script to identify and address missing or corrupted files, with no corrupted images found in the dataset.
Normalization: It includes code for normalizing image pixel values to ensure consistent scaling across the dataset.

### Labeling: The project includes a process for labeling each image as 'real' or 'manipulated'. The labeling is based on the directory name of each image, with paths and labels stored in a dictionary.

### Data Augmentation (Optional): To increase the diversity of the dataset, data augmentation techniques such as rotation, flipping, and scaling are applied. The augmented images are saved in a separate 'Augmented_Dataset' directory.

### Exploratory Data Analysis (EDA):

Visual Inspection: A sample of images is manually inspected to understand their characteristics. The notebook contains code to randomly select and display five sample images.
Class Distribution: The balance between real and manipulated images is analyzed. The dataset is confirmed to be balanced with an equal number of real and fake images (70,001 in each category) in the training, testing, and validation subsets.
Feature Selection: The notebook touches on the topic of identifying features that are most indicative of an image being manipulated. However, specific details about the feature selection process are not provided in this summary.

### Color Histogram Analysis:

The notebook includes code for generating and displaying color histograms for real and manipulated images to visually compare their color distributions. This helps in identifying differences in color patterns that could indicate image manipulation.
Exploratory Data Analysis (Continued):

Initial Findings: Although specific details are not provided in the summary, this section likely summarizes the findings from the EDA to guide the model building process.
### Model Selection and Training:

Model Selection: A Convolutional Neural Network (CNN) is chosen for classifying images into real and manipulated categories. The notebook includes details on the model's layers, output shapes, and parameters.
Test-Train Split: The notebook utilizes TensorFlow Keras' ImageDataGenerator and flow_from_directory to create training, validation, and test data generators for image classification.
CNN Training and Validation: The CNN model is trained and validated, employing techniques like checkpointing and early stopping to ensure optimal performance and avoid overfitting. The notebook tracks accuracy and loss metrics for each training epoch.
Hyperparameter Tuning:

. ### Introduction and Background
This project, titled "Deep Fake Image Detection - Real and Fake," is focused on distinguishing between real and manipulated (deep fake) images of human faces. The primary objective is to develop a reliable method for detecting deep fakes, which are becoming increasingly prevalent and sophisticated. This project is significant in the context of digital media authenticity, privacy concerns, and the prevention of misinformation.

### 2. Data Preparation and Preprocessing
The dataset, obtained from Zenodo, consists of 256x256 pixel JPG images categorized as real or manipulated. The preparatory steps included:

Data Cleaning and Inspection: Verification of image formats and sizes, ensuring consistency across the dataset.
Handling Missing/Corrupted Images: Identification and resolution of any missing or corrupted files, with the dataset found to be intact.
Normalization: Standardization of image pixel values for uniform scaling.
Labeling: Accurate categorization of images as real or manipulated based on their directory names.
Data Augmentation: Optional techniques like rotation, flipping, and scaling applied to enhance dataset diversity.
### 3. Exploratory Data Analysis (EDA)
Key EDA steps involved:

Visual Inspection: Manual examination of a sample of images to understand their characteristics.
Class Distribution: Ensuring a balanced distribution between real and manipulated images.
Color Histogram Analysis: Analyzing color distributions to identify patterns indicative of image manipulation.
Initial Findings: Summarization of EDA findings to guide the model building.
### 4. Model Selection and Training
The project employed a Convolutional Neural Network (CNN) for image classification. The process included:

CNN Configuration: Setting up a CNN with TensorFlow Keras, detailing layers, output shapes, and parameters.
Test-Train Split: Utilizing TensorFlow Keras tools for dividing the dataset into training, validation, and test sets.
Model Training and Validation: Implementing checkpointing and early stopping techniques during the training and validation phases to optimize performance and prevent overfitting.
### 5. Hyperparameter Tuning
While specific details are not provided in the summary, this phase likely involved adjusting model parameters to enhance performance.

### 6. Conclusion and Future Work
The project successfully established a structured approach to detecting deep fake images using CNNs. Key takeaways include the importance of thorough data preparation, the effectiveness of CNNs in image classification tasks, and the potential for further improvements through advanced techniques and hyperparameter optimization.

### 7. Acknowledgments
Special thanks to the providers of the dataset and the tools used in the project, including TensorFlow Keras and Python libraries like PIL and matplotlib.

### Flask Web Application: 

The app.py file appears to be a Python script for a web application using Flask, a popular micro web framework. Below is a brief explanation of the main components of the script:

Importing Libraries:

Flask-related imports for handling web requests and rendering templates.
Werkzeug utility for secure file handling.
TensorFlow Keras for loading the trained model and processing images.
NumPy for numerical operations.
OS module for handling file paths.
Warnings module to suppress warnings.
Initializing the Flask Application:

The script initializes a Flask application instance (app = Flask(__name__)).
Loading the Trained Model:

It specifies a path to the trained model (MODEL_PATH) and loads the model using Keras (load_model).
Defining Routes:

Home Route (/): A route for the main page (index() function) that renders an HTML template (index.html).
Prediction Route (/predict):
This route handles POST requests to predict the class of an uploaded image.
The uploaded file is received from the request and potentially saved to a directory.
The script includes commented code for saving the file, indicating flexibility in handling file uploads.
Prediction Logic:

The specific details of how the prediction is made are not shown in the snippet, but it likely involves processing the uploaded image and using the loaded model to make a prediction.
Running the App:

While not shown in the snippet, the script likely ends with a conditional to run the app (if __name__ == '__main__': app.run()).


