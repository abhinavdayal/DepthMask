# S15 Assignment

I applied the following strategy in attempt to solve the assignment

1. **Create a DataSet and DataLoader for our new dataset**. Since we have to experiment with small sizes initially, I implemented data loader to allow for an intended image size as output and also customize the fraction of dataset to take and also the train test split.
2. **Use Tensorboard**. Learnt to use TensorBoard with Pytorch. Had to write a separate module in library to use tensorboard to output the Training outcomes instead of json files used earlier.
3. **Separate Network from Loss criterion**. As earlier we already have create modular library, I separated the loss function from the Network definition, so loss functions can now be easily interchanged.
4. **Augmentations**: Only used brighness and contrast adjustments. Can use RGB shift, addition of sopme noise etc. Rotations, flips, cutouts etc. I thought may not be necessay in this case. Cutouts certainly may not help uch as it is not for object detection. Can try combining 4 images into one with a random crop after initial experimentation.
5. **LR Scheduler**: Initially used 10-20 epochs with One Cycle Policy using LR Finder.

The above process took good 10-12 hours to learn, debug and get it right. For example Albumentations would return gray scale images transposed, a wierd inplace error for gradient calculation asking to change x += out to x = x + out and so on.


## First Experiment

**GOAL**: Intent was to keep the network really small and see what is working and what is not. At the same time challenge was to get enough receptive field and keep the output having same dimension as the input.

**NETWORK**: Started with following network. It has around **230K parameters**. We trained on **32K images** and tested on **8K images** per epoch with a **batch size of 256**. It took around **3 minutes per epoch**. We used **L1 Loss with reduction of "sum"** instead of mean to avoid a division. **Learning rate was 0.0002** which I found with a simple LR finder over 3 epochs. **Receptive Field** is 120.

The loss came down from 300K to 230K and there were no signs of overfitting. Test loss was similar to train loss.

![DNN1](DNN1.png)

### INPUTS

![DNN1](DNN2_input.png)

![DNN1](DNN2_minput.png)

![DNN1](DNN2_dinput.png)

### Outcomes

#### TRaining Loss
![DNN1](DNN1_trainloss.png)

##### Test Loss
![DNN1](DNN1_testloss.png)

I found that the network had too much **checker board** issue and was not training that well so I updated it as below. I reduced amount of dilation in atrous convolutions and replaced transposed conv with pixel shuffle. The resulting netowrk improved further. Traned for 20 epochs with LR of 0.0002 and another 10 epochs with LR of 0.00002. Results can be seen below. The loss did not decrease beyond 125K though.

### Updated Network

I rediced dilation. **Receptive Field** is 120. Switched to pixel shuffle.

![DNN2](DNN2.png)

### Outcomes

#### TRaining Loss
![DNN1](DNN2_trainloss.png)

#### Test Loss
![DNN1](DNN2_testloss.png)

#### Results 

![DNN1](DNN2_m30.png)

![DNN1](DNN2_d30.png)

### Doubling the Features?
Tried same network as above with double the number of channels in each layer, but the outcome was not even close. My guess is that increasing the channels confused the system as the data being passed is lowres. You need more input data to work with more channels. It is just like in a Logistic Regression, adding more features is not always helpful.

## Using MSELOSS with mean and not sum
This gave a lot better results after 35 epochs of training, beyond whch it did not show promise to train further as shown below. At this point I want to try some more loss functions before trying to modify our network (encoder decoder architectures, Resnet type variants etc.) and/or the way we create the imges (e.g. thresholding, quantization, etc.)

* [LINK to ipynb](https://github.com/abhinavdayal/DepthMask/blob/master/S15Assignment_attempt1.ipynb)
* [LINK to network code](https://raw.githubusercontent.com/abhinavdayal/EVA4_LIBRARY/master/EVA4/eva4models/s15net.py)

![DNN1](MSE_trainloss.png)

![DNN1](MSE_testloss.png)

![DNN1](MSE_mask.png)

![DNN1](MSE_depth.png)

## USING SSIM Loss with window = 11 and reduction as mean
The depth calculation was much better. The MASKS however were almost blank images, so nothing to show. Notice how the checkerboard effect is gone and pixel shuffle is working way better with SSIM loss.

### Depth Output
![DEPTH](ssimdepth.jpg)
### Depth Input
![DEPTH](ssimdepthi.jpg)

## Custom loss: using MSE for mask and SSIM for Depth

Till now I represented the problem as a reconstruction (regression) problem and for those these are the best losses. So why not use what worked well for each.

```
class CustomLoss(nn.Module):
    def __init__(self, maskloss, depthloss):
        super(CustomLoss, self).__init__()
        self.maskloss = maskloss
        self.depthloss = depthloss

    def forward(self, input, target):
        maskloss = self.maskloss(input[:,:1,:,:], target[:,:1,:,:])
        depthloss = self.depthloss(input[:,1:,:,:], target[:,1:,:,:])
        return maskloss + depthloss
```

Below, pink line indicates custom loss and blue indicates only SSIM. Both were trained for 20 epochs.

![mixed](mixed_testloss.png)

![mixed](mixed_trainloss.png)

![mixed](mixed_mask.png)
![mixed](mixed_imask.png)

![mixed](mixed_depth.png)
![mixed](mixed_idepth.png)

## DICE Loss for Mask and SSIM for Depth

DICE did not work so well for mask as shown below:

![dice](dice_mask.png)

## Focal loss for Depth


## Using Classification based losses with One Hot Encoding the output and input


## Next Iteration - Deeper Network - No Atrous - Use Depthwise Separation

