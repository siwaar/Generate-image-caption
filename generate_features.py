"""Generate features from images and store them
"""
import os
import pickle
from tqdm  import tqdm
#from image_captioning.utils.loggers import setup_logger
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model



#logger, _ = setup_logger("Generate Features from images using VGG", output_folder=True)

def generate_images_features(path_to_images: str, model):
    # extract features from image
    features = {}
    images_names = os.listdir(path_to_images)
    for img_name in tqdm(images_names):
        # load the image from file
        img_path = path_to_images + '/' + img_name
        image = load_img(img_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for vgg
        image = preprocess_input(image)
        # extract features
        feature = model.predict(image, verbose=0)
        # get image ID
        image_id = img_name.split('.')[0]
        # store feature
        features[image_id] = feature
        return features


if __name__ =="__main__":
    # load vgg16 model
    model = VGG16()
    # restructure the model
    VGG_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    #logger.info(model.summary())
    directory = os.path.join('data/flickr8k/images')
    #logger.info('Generate image features')
    features = generate_images_features(path_to_images=directory, model=VGG_model)
    # store features in pickle
    #logger.info('Save image features in pickle')
    print('a')
    path_to_save_features = 'data/features.pkl'
    pickle.dump(features, open(os.path.join(path_to_save_features), 'wb'))
    #logger.info(f'Generated features stored in {path_to_save_features}')