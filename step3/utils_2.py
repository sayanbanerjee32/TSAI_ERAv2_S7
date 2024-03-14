from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

import pandas as pd

SEED = 1
## function to check if GPU is available and return relevant device
def get_device():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    return cuda, torch.device("cuda" if cuda else "cpu")    

## Return prediction count based on model prediction and target
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

### train function that will be called for each epoch
def train(model, device, train_loader, optimizer, criterion):
  # this only tells the model that training will start so that training related configuration can be swithed on
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # move all data to same device
    data, target = data.to(device), target.to(device)
    # No gradient accumulation
    optimizer.zero_grad()

    # Predict - this calls the forward function
    pred = model(data)

    # Calculate loss - 
    # difference between molde prediction and target values based on criterion function provided as input
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    # takes a step, i.e. updates parameter values based on gradients
    # these parameter values will be used for the next batch
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    # Shows traing progress with loss and accuracy update for each batch
    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

#   train_acc.append(100*correct/processed)
#   train_losses.append(train_loss/len(train_loader))
  # returns training accuracy and loss for the epoch  
  return (100*correct/processed), (train_loss/len(train_loader))

### test function that will be called for each epoch
def test(model, device, test_loader, criterion):
    # switching on eval / test mode
    model.eval()

    test_loss = 0
    correct = 0
    # no gradient calculation is required for test
    ## as parameters are not updated while test
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Calculate test loss - 
            # difference between molde prediction and target values based on criterion function provided as input
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    # test_acc.append(100. * correct / len(test_loader.dataset))
    # test_losses.append(test_loss)

    # print test loss and accuracy after each epoch
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # returns test accuracy and loss for the epoch
    return (100. * correct / len(test_loader.dataset)), test_loss

# function to plot train and test accuracies and losses
def plot_accuracy_losses(train_losses, train_acc, test_losses, test_acc, num_epochs):
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    ax = plt.gca()
    ax.set_xlim([0, num_epochs + 1])
    plt.ylabel('Loss')
    plt.plot(range(1, num_epochs + 1),
             train_losses[:num_epochs+1],
              'r', label='Training Loss')
    plt.plot(range(1, num_epochs + 1),
             test_losses[:num_epochs+1],
             'b', label='Test Loss')
    ax.grid(linestyle='-.')
    plt.legend()
    plt.subplot(2,1,2)
    ax = plt.gca()
    ax.set_xlim([0, num_epochs+1])
    plt.ylabel('Accuracy')
    plt.plot(range(1, num_epochs + 1),
             train_acc[:num_epochs+1],
              'r', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1),
             test_acc[:num_epochs+1],
              'b', label='Test Accuracy')
    ax.grid(linestyle='-.')
    plt.legend()
    plt.show()
  

# get predicted value based on argmax
def GetPrediction(pPrediction):
  return pPrediction.argmax(dim=1)

# Returns individual image with target, prediction and loss
# after batch inference
def get_individual_loss(model, device, data_loader, criterion):
    # switching on eval / test mode
    model.eval()

    loss_list = []


    # no gradient calculation is required for test
    ## as parameters are not updated while test
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            for d,t,p,l in zip(data, target,
                               GetPrediction(output),
                               criterion(output, target, reduction='none')):
                loss_list.append((d[0].to('cpu'),
                                  t.to('cpu').item(),
                                  p.to('cpu').item(),
                                  l.to('cpu').item()))


    return loss_list
# the following function will plot images with target and predicted values
# where prediction is wrong - this will be in order or decreasing loss
# group by target
def plot_top_loss(model, device, data_loader, criterion, max_item = 5):
    loss_list = get_individual_loss(model, device, data_loader, criterion)
    loss_df = pd.DataFrame(loss_list, columns=['image', 'target', 'prediction', 'loss'])
    # incorrect
    incorrect_df = loss_df[loss_df.prediction != loss_df.target]
    print(f"total wrong predictions: {incorrect_df.shape[0]}")

    incr_groups = incorrect_df.groupby(['target','prediction']).agg({'loss':'median',
                                                             'image':'count'}).reset_index().sort_values(by='image', ascending=False)

    incorrect_df = incorrect_df.sort_values(by='loss', ascending=False)
    
    for t in incorrect_df.target.unique():
        t_img_df = incorrect_df.loc[incorrect_df.target == t].copy()
        num_img = max_item if t_img_df.shape[0] > max_item else t_img_df.shape[0]
        print(f'target: {t}, num_img: {num_img}')
        for i, row in t_img_df.head(num_img).iterrows():
            plt.figure(figsize=(2, 2))
            plt.imshow(row['image'].numpy().reshape(28,28), cmap='gray')
            plt.title(f'target: {row["target"]}, prediction: {row["prediction"]}, loss: {round(row["loss"],2)}')
            plt.axis('off')
            plt.show()

    return incr_groups
    
