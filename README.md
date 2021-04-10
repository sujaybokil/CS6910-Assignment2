# CS6910-Assignment2
Assignment 2 of the CS6910: Fundamentals of Deep Learning course by Sujay Bokil (ME17B120) and Avyay Rao (ME17B130)

## Part A

1. The notebook is structured to be run cell by cell.

2. Next, the google drive needs to be mounted and the iNaturalist file needs to be unzipped. This part of the code will need to be modified according to the filepath on your local machine.

```python
#Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Load and unzip iNaturalist zip file onto server, then remove zip to optimize performance
zip_path = "drive/MyDrive/nature_12K.zip"
!cp "{zip_path}" .
!unzip -q nature_12K.zip
!rm nature_12K.zip
```
3. There are functions defined to build a custom CNN and to prepare the image data generators for training and testing which need to be compiled.

4. Functions train() and test() integrate WandB with the training, validation and testing process. A sweep config is defined already, whose hyperparameters and values can be modified. The train or test function can be called by the sweep agent.

```python
sweep_config = {
    "name": "Test and Save Best Model",
    "description": "Checking the performance of CNN on sample of test data",
    "metric": "Val Accuracy",
    "method": "grid",
    "project": "CS6910_Assignment2",
    "parameters": {
        "num_filters": {
            "values": [32]
        },
        "filter_multiplier": {
            "values": [2]
        },
        "augment_data": {
            "values": [True]
        },
        "dropout": {
            "values": [0.3]
        },
        "batch_norm": {
            "values": [True]
        },
        "epochs": {
            "values": [1]
        },
        "dense_size": {
            "values": [64]
        },
        "lr": {
            "values": [0.001]
        },
        "epochs": {
            "values": [10]
        }
    }
}

# creating the sweep
sweep_id = wandb.sweep(sweep_config, project="CS6910_Assignment2")
```

5. Further, there are functions provided to plot sample image predictions and filter visualizations on the test data, which can be run from within the test function.

6. Also, there is a function which can customise the run names in WandB.

## Part B

1. The notebook is structured such that it can be ran cell by cell

2. For training the model without wandb, use the following code snippet

```python
# Data loading
train_ds, val_ds = load_data((256, 256))

# Instantiate a model (InceptionV2, InceptionResNetV2, ResNet50, Xception)
model = NeuralNet("InceptionV3", image_shape=(256,256))

# Compile the model 
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy']) 

# Fitting and validating the model
model.fit(train_ds, 
            validation_data=val_ds,
            epochs=30)
```

3. For training the model and fine-tuning with wandb, we have created a `train_with_wandb` function

```python
## Setting up the sweep ##
sweep_config = {
  "name": "My Sweep",
  "method": "grid",
  "parameters": {
        "base_model": {
            "values": ["InceptionV3", "InceptionResNetV2", "ResNet50", "Xception"]
        }
    }
}

# Creating a sweep
sweep_id = wandb.sweep(sweep_config)

# Running the sweep
wandb.agent(sweep_id, function=train_with_wandb)
```

## Part C

Pre-requisites: OpenCV (use the commands below to install)

`pip install opencv` or `conda install -c conda-forge opencv`

1. Download the weights for the mask detection demo from `PartC/weights.md` and put it inside the `PartC/mask_weights` directory

2. Run the following to change the present directory to the PartC directory

```
cd /path/to/repo/PartC
```

3. To run the demo on the image, run the follwing on a console

```
python main.py --image_path "/path/to/image"
````

Below is the code to run it on one of the demo images

```
python main.py --image_path "images/bad_mask"
```

3. To run it on your webcam, run the following on your console

```
python main.py --webcam True
```
