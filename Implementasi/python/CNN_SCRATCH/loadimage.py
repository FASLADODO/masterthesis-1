import dataset

#Prepare input data
classes = ['dogs','cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 50
num_channels = 1
train_path='training_data'
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
