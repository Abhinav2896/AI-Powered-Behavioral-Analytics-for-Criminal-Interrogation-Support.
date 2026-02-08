# # from tensorflow.keras.preprocessing.image import ImageDataGenerator
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# # from tensorflow.keras.models import load_model
# # import os



# # train_data_dir='data/train/'
# # validation_data_dir='data/test/'


# # train_datagen = ImageDataGenerator(
# # 					rescale=1./255,
# # 					rotation_range=30,
# # 					shear_range=0.3,
# # 					zoom_range=0.3,
# # 					horizontal_flip=True,
# # 					fill_mode='nearest')

# # validation_datagen = ImageDataGenerator(rescale=1./255)

# # train_generator = train_datagen.flow_from_directory(
# # 					train_data_dir,
# # 					color_mode='grayscale',
# # 					target_size=(48, 48),
# # 					batch_size=32,
# # 					class_mode='categorical',
# # 					shuffle=True)

# # validation_generator = validation_datagen.flow_from_directory(
# # 							validation_data_dir,
# # 							color_mode='grayscale',
# # 							target_size=(48, 48),
# # 							batch_size=32,
# # 							class_mode='categorical',
# # 							shuffle=True)


# # class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# # img, label = train_generator.__next__()


# # model = Sequential()

# # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

# # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.1))

# # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.1))

# # model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.1))

# # model.add(Flatten())
# # model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.2))

# # model.add(Dense(7, activation='softmax'))

# # model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # print(model.summary())


# # train_path = "data/train/"
# # test_path = "data/test"

# # num_train_imgs = 0
# # for root, dirs, files in os.walk(train_path):
# #     num_train_imgs += len(files)
    
# # num_test_imgs = 0
# # for root, dirs, files in os.walk(test_path):
# #     num_test_imgs += len(files)

# # print(num_train_imgs)
# # print(num_test_imgs)
# # epochs=1

# # history=model.fit(train_generator,
# #                 steps_per_epoch=num_train_imgs//32,
# #                 epochs=epochs,
# #                 validation_data=validation_generator,
# #                 validation_steps=num_test_imgs//32)

# # model.save('model_file.h5')






# #new code
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
import os

# Paths
train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Count images
num_train_imgs = sum(len(files) for _, _, files in os.walk(train_data_dir))
num_test_imgs = sum(len(files) for _, _, files in os.walk(validation_data_dir))

print(f"Train images: {num_train_imgs}")
print(f"Test images: {num_test_imgs}")

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train
epochs = 101
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // 32,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // 32,
    callbacks=[early_stop]
)

# Save model
model.save('model_file.h5')

# Save training history
with open("training_history.txt", "w") as f:
    for key in history.history:
        f.write(f"{key}: {history.history[key]}\n")