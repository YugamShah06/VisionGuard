# import numpy as np
# from matplotlib import image
# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
# from tensorflow.keras.applications import VGG16

# class Predictions:
    
#     def __init__(self, model=False):
#         # attribute showing whether model weights have been provided or generated
#         self.loaded_model = model
#         self.model = model
#         self.loaded_image = False
#         self.loaded_image_features = False
        
#     def load_image(self, filename, save_image=True):
       
#         # Load the image
#         img = load_img(filename, target_size=(224, 224))
#         # Convert to array
#         img = img_to_array(img)
#         # Reshape into a single sample with 3 channels
#         img = img.reshape(1, 224, 224, 3)
#         # Center pixel data
#         img = img.astype('float32')
#         img -= [123.68, 116.779, 103.939]
        
#         if save_image:
#             self.filename = filename
#             self.image = img
#             self.loaded_image = True

#         return img
    
#     def get_bottleneck_features(self, save_features=True):
        
#         if not self.loaded_image:
#             raise ValueError('Must load an image before generating features. Call the load_image method and set save_image to True')
        
#         # Instantiate VGG16 model
#         model = VGG16(include_top=False, input_shape=(224, 224, 3))

#         # Run image through VGG16 and store features
#         image_features = model.predict(self.image)

#         if save_features:
#             self.image_features = image_features
#             self.loaded_image_features = True

#         return image_features
    
#     def load_model(self, weights_filepath):

        
#         # Create model architecture using Keras Functional API
#         inputs = Input(shape=(7, 7, 512))
#         flat = Flatten()(inputs)
#         class1 = Dense(256, activation='relu')(flat)
#         drop1 = Dropout(0.3)(class1)
#         class2 = Dense(128, activation='relu')(drop1)
#         drop2 = Dropout(0.65)(class2)
#         class3 = Dense(64, activation='relu')(drop2)
#         drop3 = Dropout(0.15)(class3)
#         output = Dense(1, activation='sigmoid')(drop3)

#         model = keras.Model(inputs=inputs, outputs=output)

#         # Set optimizer to stochastic gradient descent
#         opt = SGD(learning_rate=0.001, momentum=0.9)
#         model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#         # Load weights
#         model.load_weights(weights_filepath)
#         self.model = model
#         self.loaded_model = True

    
#     # def predict(self, filename, weights_filepath):
        
#     #     img = self.load_image(filename)
#     #     image_features = self.get_bottleneck_features()
        
#     #     # Load model weights if not already loaded
#     #     if not self.loaded_model:
#     #         self.load_model(weights_filepath)
        
#     #     # Get prediction
#     #     self.prediction = self.model(self.image_features, training=False).numpy()[0][0]

#     #     # Print prediction result
#     #     if self.prediction > 0.5:
#     #         print('Healthy')
#     #     else:
#     #         print('Conjunctivitis - consider seeing a medical professional')
#     #     return self.prediction

#     def analyze_redness(self, threshold=150):

#         if not self.loaded_image:
#             raise ValueError('Must load an image before analyzing redness.')

#         # Extract the red channel from the image
#         red_channel = self.image[0, :, :, 0]  # Since the image is already in (1, 224, 224, 3) shape

#         # Count the number of red pixels above the threshold
#         redness_level = np.sum(red_channel > threshold)

#         # Normalize by total number of pixels in the image
#         total_pixels = red_channel.shape[0] * red_channel.shape[1]
#         redness_ratio = redness_level / total_pixels

#     # Return True if redness is above a reasonable threshold
#         return redness_ratio > 0.3  # You can tune this value as per your data


#     def predict(self, filename):


#         # Calls load_image method to generate 224 x 224 x 3 array
#         img = self.load_image(filename)
#         # Calls get_bottleneck_features method to run input image through VGG16
#         image_features = self.get_bottleneck_features()
#         # Calls load_model method if predict is being run for the first time
#         if not self.loaded_model:
#             self.load_model()

#         # Get prediction from the model
#         self.prediction = self.model(self.image_features, training=False).numpy()[0][0]

#         # Check if significant redness is detected
#         redness_detected = self.analyze_redness()

#         # Print the prediction result
#         if self.prediction > 0.5:
#             print('Healthy')
#         else:
#             if redness_detected:
#                 print('Serious Conjunctivitis - immediate medical attention recommended.')
#             else:
#                 print('Conjunctivitis - consider seeing a medical professional.')

#         return self.prediction
    
#     def show_image(self):
#         '''Displays image contained in self.image'''
#         if not self.loaded_image:
#             raise ValueError('Must load an image before printing. Call the load_image method and set save_image to True')
#         data = image.imread(self.filename)
#         plt.imshow(data)
#         plt.axis('off')  # Hide axes
#         plt.show()


# # Create an instance of Predictions
# prediction_instance = Predictions()

# # Specify the paths for the weights file and the image file
# weights_file_path = "weights/model-pink-eye-weights-88-0.942.hdf5"  # Change this to your weights path
# image_file_path = "scraping/conjunctivitis bing/C0111797-Adenoviral_conjunctivitis_of_the_eyes.jpg"  # Change this to your image path

# # Make a prediction
# prediction_instance.predict(image_file_path, weights_file_path)

# # Optionally, show the image
# prediction_instance.show_image()



import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.applications import VGG16

class Predictions:
    
    def __init__(self, model=False):
        # attribute showing whether model weights have been provided or generated
        self.loaded_model = model
        self.model = model
        self.loaded_image = False
        self.loaded_image_features = False
        
    def load_image(self, filename, save_image=True):
        '''Loads an image, processes it, and stores it in the instance'''
        # Load the image
        img = load_img(filename, target_size=(224, 224))
        # Convert to array
        img = img_to_array(img)
        # Reshape into a single sample with 3 channels
        img = img.reshape(1, 224, 224, 3)
        # Center pixel data
        img = img.astype('float32')
        img -= [123.68, 116.779, 103.939]
        
        if save_image:
            self.filename = filename
            self.image = img
            self.loaded_image = True

        return img
    
    def get_bottleneck_features(self, save_features=True):
        '''Generates bottleneck features from VGG16'''
        if not self.loaded_image:
            raise ValueError('Must load an image before generating features. Call the load_image method and set save_image to True')
        
        # Instantiate VGG16 model
        model = VGG16(include_top=False, input_shape=(224, 224, 3))

        # Run image through VGG16 and store features
        image_features = model.predict(self.image)

        if save_features:
            self.image_features = image_features
            self.loaded_image_features = True

        return image_features
    
    def load_model(self, weights_filepath):
        '''Creates and loads model weights'''
        # Create model architecture using Keras Functional API
        inputs = Input(shape=(7, 7, 512))
        flat = Flatten()(inputs)
        class1 = Dense(256, activation='relu')(flat)
        drop1 = Dropout(0.3)(class1)
        class2 = Dense(128, activation='relu')(drop1)
        drop2 = Dropout(0.65)(class2)
        class3 = Dense(64, activation='relu')(drop2)
        drop3 = Dropout(0.15)(class3)
        output = Dense(1, activation='sigmoid')(drop3)

        model = keras.Model(inputs=inputs, outputs=output)

        # Set optimizer to stochastic gradient descent
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Load weights
        model.load_weights(weights_filepath)
        self.model = model
        self.loaded_model = True

    def analyze_redness(self, threshold=85):
        '''Analyzes the level of redness in the image'''
        if not self.loaded_image:
            raise ValueError('Must load an image before analyzing redness.')

        # Extract the red channel from the image
        red_channel = self.image[0, :, :, 0]  # Since the image is already in (1, 224, 224, 3) shape

        # Count the number of red pixels above the threshold
        redness_level = np.sum(red_channel > threshold)

        # Normalize by total number of pixels in the image
        total_pixels = red_channel.shape[0] * red_channel.shape[1]
        redness_ratio = redness_level / total_pixels

        # Return True if redness is above a reasonable threshold
        return redness_ratio > 0.3  # You can tune this value as per your data

    def predict(self, filename, weights_filepath):
        '''Makes predictions on an input image with redness analysis'''
        # Calls load_image method to generate 224 x 224 x 3 array
        img = self.load_image(filename)
        # Calls get_bottleneck_features method to run input image through VGG16
        image_features = self.get_bottleneck_features()

        # Load model weights if not already loaded
        if not self.loaded_model:
            self.load_model(weights_filepath)

        # Get prediction from the model
        self.prediction = self.model(self.image_features, training=False).numpy()[0][0]

        # Check if significant redness is detected
        redness_detected = self.analyze_redness()

        # Print the prediction result
        if self.prediction > 0.5:
            print('Healthy')
        else:
            if redness_detected:
                print('Serious Conjunctivitis - immediate medical attention recommended.')
            else:
                print('Conjunctivitis - consider seeing a medical professional.')

        return self.prediction
    
    def show_image(self):
        '''Displays the loaded image'''
        if not self.loaded_image:
            raise ValueError('Must load an image before printing. Call the load_image method and set save_image to True')
        data = image.imread(self.filename)
        plt.imshow(data)
        plt.axis('off')  # Hide axes
        plt.show()


# Create an instance of Predictions
prediction_instance = Predictions()

# Specify the paths for the weights file and the image file
weights_file_path = "weights/model-pink-eye-weights-88-0.942.hdf5"  # Change this to your weights path
image_file_path = "eye-condition-detection-deep-learning-main/scraping/infection/blog-eye-care-1.jpg"  # Change this to your image path

# Make a prediction
prediction_instance.predict(image_file_path, weights_file_path)

# Optionally, show the image
prediction_instance.show_image()

# scraping\conjunctivitis bing\C0111797-Adenoviral_conjunctivitis_of_the_eyes.jpg
