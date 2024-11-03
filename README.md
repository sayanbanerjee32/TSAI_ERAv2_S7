# Vision Modelling steps

## Objective
- Achieving test accuracy of 99.4% (this must be consistently shown in last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 8000 Parameters
- Achieving this in 3 or 4 steps

## Steps

The assignment is done in 4 steps including the initial step to create the basic structure of code and model skeleton for training and testing iteratively.

### Step 0 - Basic set-up
[Go to the Step 0 notebook](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/blob/main/step0/S7_step0_SayanBanerjee.ipynb)  
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
[Go to the Step 1 notebook](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/blob/main/step1/S7_step1_SayanBanerjee.ipynb)  
#### Target:
- Make the model lighter - as much as possible
- Add Batch-norm to increase model efficiency.

#### Results:
- Parameters: 3252
- Best Training Accuracy: 98.53%
- Best Test Accuracy: 98.35%

#### Analysis:
- Most of epochs have shown slight overfitting. Adding Drop-out might help reducing overfitting.
- Train and test accuracy and losses have plateaued in later epochs.
- If model capacity increased (i.e. increased number of parameters), the model has capability to achieve 99.4% accuracy on test set.

### Step 2 - Regularise and add capacity
[Go to the Step 2 notebook](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/blob/main/step2/S7_step2_SayanBanerjee.ipynb)  
#### Target:
- Add Regularization - Dropout
- Start transition block at receptive field 5
- Increase model capacity. Add more layers at the end.

#### Results:
- Parameters: 7432
- Best Training Accuracy: 98.83%
- Best Test Accuracy: 99.07%

#### Analysis:
- Model does not show any overfitting, however, able to reach 99% accuracy consistently
- Test loss is fluctuating towards end epochs. Thus use of LR scheduler might be helpful
- Error analysis suggests that there are combinations where model is confused more than others. e.g. target 9-predicted 4, target 2-predicted 7, target 8-predicted 6 are the top 3 cases with respect to number of errors, while target 6-predicted 4, target 9-predicted 7, target 3-predicted 7 are the top 3 cases with respect to median losses. Adding image augmentation for training set might be helpful in pushing Test accuracy further.  
See examples below:  
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/be7dba95-1411-4cc7-ab42-66fee6b086fc)
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/9166d954-310e-44f7-ae4b-0d5a1848166b)
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/5881f741-f992-498e-93b6-7b595d2831be)
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/11b0ae44-726a-4c4b-9a49-b57556abd614)


### Step 3 - Image Augmentation and Learning rate tuning
[Go to the Step 3 notebook](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/blob/main/step3/S7_step3_SayanBanerjee.ipynb)  
#### Target:
- Add image augmentation and trial with different combinations of augmentation to reduce errors identified in the last step
- Add learning rate scheduler so that learning rate can be decreased when the loss gets plateaued

#### Results:
- Parameters: 7432
- Best Training Accuracy: 98.65%
- Best Test Accuracy: 99.51%

#### Analysis:
- Started with high learning rate so that loss gets reduced quickly in initial epochs. Learning rate is reduced at a later point using ReduceLROnPlateau for controlled convergence of loss.
- Image augmentation helped in increasing test accuracy compared to train accuracy significantly.
- Comparison of error analysis between last and this step shows that most of tor error are reduced except there are still 5 cases of 7 being predicted as 1. On closer look of the data, it seems difficult even for a person to make those recognition correctly due to the nature of the data.  
See examples below:  
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/df74fffd-76b6-4465-b31a-55218f6b4ad3)
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S7/assets/11560595/c8c22234-774b-4355-812e-466d64175672)

