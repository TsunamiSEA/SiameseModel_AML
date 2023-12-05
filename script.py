import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
# from tensorflow.keras.applications import resnet_v2
# from tensorflow.keras.applications.resnet_v2 import preprocess_input
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import backend, layers, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay

# Setting random seeds to enable consistency while testing.
random.seed(5)
np.random.seed(5)
tf.random.set_seed(5)

ROOT = "C:/Users/101234758/Desktop/MLP/data"

def read_image(index):
    path = os.path.join(ROOT, index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    return image

def split_dataset(directory, split=0.8):
    folders = os.listdir(directory)
    num_train = int(len(folders)*split)
    
    random.shuffle(folders)
    
    train_list, test_list = {}, {}
    
    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files
    
    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files  
    
    return train_list, test_list

train_list, test_list = split_dataset(ROOT, split=0.8)
print("Length of training list:", len(train_list))
print("Length of testing list :", len(test_list))

# train_list, test list contains the folder names along with the number of files in the folder.
# print("\nTest List:", test_list)

def create_triplets(directory, folder_list, max_files=2):
    triplets = []
    folders = list(folder_list.keys())
    
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)
        
        for i in range(num_files-1):
            for j in range(i+1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(0, folder_list[neg_folder]-1)
                negative = (neg_folder, f"{neg_file}.jpg")

                triplets.append((anchor, positive, negative))
            
    random.shuffle(triplets)
    return triplets

train_triplet = create_triplets(ROOT, train_list)
test_triplet  = create_triplets(ROOT, test_list)

print("Number of training triplets:", len(train_triplet))
print("Number of testing triplets :", len(test_triplet))

print("\nExamples of triplets:")
for i in range(5):
    print(train_triplet[i])
    
def get_batch(triplet_list, batch_size=32, preprocess=True):
    batch_steps = len(triplet_list) // batch_size

    for i in range(batch_steps + 1):
        anchor = []
        positive = []
        negative = []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            current_anchor = read_image(a)
            current_positive = read_image(p)
            current_negative = read_image(n)

            anchor.append(current_anchor)
            positive.append(current_positive)
            negative.append(current_negative)
            j += 1

        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)

        if preprocess:
            # Check if any of the shapes is (0, ...) during preprocessing
            if anchor.shape[0] == 0 or positive.shape[0] == 0 or negative.shape[0] == 0:
                # print(f"Skipping iteration {j} due to empty image(s) during preprocessing.")
                continue

            anchor = preprocess_input(anchor)
            positive = preprocess_input(positive)
            negative = preprocess_input(negative)

            # # Print shapes after preprocessing
            # print("After Preprocessing:")
            # print("Anchor shape:", anchor.shape)
            # print("Positive shape:", positive.shape)
            # print("Negative shape:", negative.shape)

        yield ([anchor, positive, negative])

        
num_plots = 3
f, axes = plt.subplots(num_plots, 3, figsize=(10,10))

# Assuming `get_batch` returns a batch of triplets in the format (anchor, positive, negative)
for x in get_batch(train_triplet, batch_size=num_plots, preprocess=False):
    a, p, n = x
    for i in range(num_plots):
        # Display Anchor image
        axes[i, 0].imshow(a[i])
        axes[i, 0].set_title('Anchor', fontsize=8) 
        axes[i, 0].axis('on')

        # Display Positive image
        axes[i, 1].imshow(p[i])
        axes[i, 1].set_title('Positive', fontsize=8)
        axes[i, 1].axis('on')

        # Display Negative image
        axes[i, 2].imshow(n[i])
        axes[i, 2].set_title('Negative', fontsize=8) 
        axes[i, 2].axis('on')
    break

def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = resnet.ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
    )
    
    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model

class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    

def get_siamese_network(input_shape = (224, 224, 3)):
    encoder = get_encoder(input_shape)
    
    # Input Layers for the images
    anchor_input   = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")
    
    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)
    
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )
    
    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )
    return siamese_network

siamese_network = get_siamese_network()
siamese_network.summary()

class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        
        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]
    
siamese_model = SiameseModel(siamese_network)

LEARNING_RATE = 0.0001
optimizer = Adam(learning_rate=LEARNING_RATE)
siamese_model.compile(optimizer=optimizer)

# Inside your test_on_triplets function
def test_on_triplets(batch_size=256):
    pos_scores, neg_scores, losses = [], [], []

    for data in get_batch(test_triplet, batch_size=batch_size):
        loss = siamese_model.test_on_batch(data)
        losses.append(loss)
        
        predictions = siamese_model.predict(data)
        pos_scores += list(predictions[0])
        neg_scores += list(predictions[1])
    
    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean = np.mean(pos_scores)
    an_mean = np.mean(neg_scores)
    ap_stds = np.std(pos_scores)
    an_stds = np.std(neg_scores)
    test_loss = np.mean(losses)
    
    print(f"\nAccuracy on test = {accuracy:.5f}")
    print(f"Loss on test      = {test_loss:.5f}")

    return (accuracy, ap_mean, an_mean, ap_stds, an_stds, test_loss)

save_all = False
epochs = 30
batch_size = 64

max_acc = 0
train_loss = []
train_accuracy = []
test_metrics = []

for epoch in range(1, epochs + 1):
    t = time.time()
    
    # Training the model on train data
    epoch_loss = []
    correct_predictions = 0
    total_samples = 0
    
    for data in get_batch(train_triplet, batch_size=batch_size):
        loss = siamese_model.train_on_batch(data)
        epoch_loss.append(loss)
        
        # Calculate training accuracy
        predictions = siamese_model.predict(data)
        pos_scores = predictions[0]
        neg_scores = predictions[1]
        
        correct_predictions += np.sum(pos_scores < neg_scores)
        total_samples += len(pos_scores)
    
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)
    
    # Calculate training accuracy
    accuracy = correct_predictions / total_samples
    train_accuracy.append(accuracy)

    print(f"\nEPOCH: {epoch} \t (Epoch done in {int(time.time()-t)} sec)")
    print(f"Accuracy on train = {accuracy:.5f}")
    print(f"Loss on train    = {epoch_loss:.5f}")

    # Testing the model on test data
    metric = test_on_triplets(batch_size=batch_size)
    test_metrics.append(metric)
    test_accuracy = metric[0]
    
    # Saving the model weights
    if save_all or test_accuracy >= max_acc:
        siamese_model.save("siamese_model", save_format="tf")
        max_acc = test_accuracy

# Saving the model after all epochs run
siamese_model.save("siamese_model-final", save_format="tf")

def plot_training_results(train_loss, train_accuracy, test_metrics):
    # Plotting Training and Testing Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', marker='o')
    plt.plot([metric[5] for metric in test_metrics], label='Testing Loss', marker='o')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Training and Testing Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy', marker='o')
    plt.plot([metric[0] for metric in test_metrics], label='Testing Accuracy', marker='o')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_results(train_loss, train_accuracy, test_metrics)

# ----------------------------------------------------------------------------- #

def extract_encoder(model):
    encoder = get_encoder((224, 224, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

encoder = extract_encoder(siamese_model)
# encoder.save_weights("encoder")
encoder.summary()

def load_validation_data(validation_file_path):
    with open(validation_file_path, 'r') as file:
        lines = file.readlines()

    image_pairs = []
    labels = []

    for line in lines:
        # Split each line into image paths and label
        parts = line.strip().split(' ')
        image_pairs.append((parts[0], parts[1]))
        labels.append(int(parts[2]))

    return image_pairs, labels

validation_file_path = "C:/Users/101234758/Desktop/MLP/verification_pairs_val.txt"
validation_image_pairs, validation_labels = load_validation_data(validation_file_path)

print(f"Number of validation samples: {len(validation_image_pairs)}")
print(f"Sample image pair: {validation_image_pairs[0]}, Label: {validation_labels[0]}")

from sklearn.metrics.pairwise import cosine_similarity

def read_image_pairs(image_pairs):
    pair_images = []
    for pair in image_pairs:  # Read only the first 100 pairs
        image1_path, image2_path = pair
        image1 = cv2.imread(image1_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, (224, 224))

        image2 = cv2.imread(image2_path)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = cv2.resize(image2, (224, 224))

        pair_images.append((image1, image2))
    return pair_images

validation_image_pairs_2 = read_image_pairs(validation_image_pairs)

def compute_cosine_similarity(encoder, image_pairs):
    similarities = []

    for pair in image_pairs:
        # Encode images using the provided encoder
        encoding1 = encoder.predict(np.expand_dims(pair[0], axis=0))
        encoding2 = encoder.predict(np.expand_dims(pair[1], axis=0))

        # Compute cosine similarity
        similarity = cosine_similarity(encoding1, encoding2)[0][0]
        similarities.append(similarity)

    return similarities

similarities = compute_cosine_similarity(encoder, validation_image_pairs_2)

threshold = 0.5  # Adjust as needed
predictions = [1 if sim > threshold else 0 for sim in similarities]

for i in range(10):
    image_pair = validation_image_pairs[i]
    actual_label = validation_labels[i]
    predicted_label = predictions[i]

    print(f"Image Pair: {image_pair}, Actual Label: {actual_label}, Predicted Label: {predicted_label}")
    
# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(validation_labels, similarities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Compute confusion matrix
cm = confusion_matrix(validation_labels, predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Different Person', 'Same Person'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
class_report = classification_report(validation_labels, predictions, target_names=['Different Person', 'Same Person'])
print("Classification Report:")
print(class_report)
