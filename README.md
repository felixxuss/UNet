# UNet implementation

This is an implementation of the UNet architecture for image segmentation. (https://arxiv.org/abs/1505.04597)

The code comes with two different ways of loading data into the model. First by using a saved dataset in a directory. Second by using a dataset provided by ```torchvision.datasets```.

## Command line arguments

- --device
  - specifies the device the model should use
  - cpu or cuda
- --data_path
  - specifies the path to the localy saved data
  - default = data/CityScapes
- --subset
  - whether the subset should be created of size --subset_size
  - default = False
- -- subset_size
  - the size of the used subset
  - default = 16
- --num_classes
  - The number of different classes in the segmentation map
  - default = 3
- --exp_name
  - The name of the current experiment. Is used in the name of the saved model
  - default = UNet
- --verbose
  - Whether the model should print stats during training
  - default = False
- --show_model
  - Whether the model is shown. The number of parameters for every layer and in total is also calculated
  - default = False
- --save_models
  - Whether the current best model should be saved
  - default = False
- --epochs
  - Number of epochs the model is trained
  - default = 5
- --batch_size
  - Size of the batches during training
  - default = 32
- --learning_rate
  - Used learning rate for the adam optimizer
  - default = 3e-4

