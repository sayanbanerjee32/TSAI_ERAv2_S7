# TSAI_ERAv2_S7

## Objective
- Achieving test accuracy of 99.4% (this must be consistently shown in last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Do this in 3/4 steps

## Steps

The assignment is doen in 4 steps including first step to create the basic structure of code and model skeleton for training and testing

### Step 0 - Basic set-up

#### Target:

- Complete initial set-up
  - Set Transforms
  - Set Data Loader
  - Set Basic Working Code
  - Set Basic Training and Test Loop

- Create the basic skeleton.

#### Results:

- Parameters: 189984
- Best Training Accuracy: 99.20%
- Best Test Accuracy: 99.02%

#### Analysis:

- A Basic skeleton is created that can train and test the model
- Model is suffering from overfitting, specially towards later epochs, Adding regularisation and batch normalisation might help
- Model is too large with respected to required number of parameters. However, there is enough room available for test accuracy to reach 99.4% if training accuracy can be improved further.


### Step 1 - Lighter model with BatchNorm


### Step 2 - Regularise and add capacity


### Step 3 - Image Augmentation and Learning rate tuning
