# S15 Assignment

I applied the following strategy in attempt to solve the assignment

1. **Create a DataSet and DataLoader for our new dataset**. Since we have to experiment with small sizes initially, I implemented data loader to allow for an intended image size as output and also customize the fraction of dataset to take and also the train test split.
2. **Use Tensorboard**. Learnt to use TensorBoard with Pytorch. Had to write a separate module in library to use tensorboard to output the Training outcomes instead of json files used earlier.
3. **Separate Network from Loss criterion**. As earlier we already have create modular library, I separated the loss function from the Network definition, so loss functions can now be easily interchanged.

The above process took good 10-12 hours to learn, debug and get it right. For example Albumentations would return gray scale images transposed, a wierd inplace error for gradient calculation asking to change x += out to x = x + out and so on.


## First Experiment

**GOAL**: Intent was to keep the network really small and see what is working and what is not. At the same time challenge was to get enough receptive field and keep the output having same dimension as the input.

**NETWORK**: Started with following network

![DNN1](DNN1.jpg)

### Updated Network

![DNN2](DNN2.jpg)
