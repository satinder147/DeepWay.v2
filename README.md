# DEEPWAY
Autonomous navigation for blind people
# Steps
1. Collecting Training data.
2. Labelling training data (used LabelBox for it).
3. Downlading data from the exported .csv file from Labelbox
4. Training a segmentation model on the data
5. 

# Things to acheieve:
1. Collecting data
2. training a simple classifier
3. generating data for segmentation of roads and people
4. training UNET
5. running everything on nano
6. Generating lines(making your own lanes
7. pushing the person the leftmost lane
8. AAgar admi agle lane me aata hai to lane change
9. Navigation using GPS module


# Hardware requirements
1. Nvidia Jetson Nano.
2. Arduino nano.
3. 2 servo motors.
4. USB audio adapter(as jetson nano does not have a audio jack)
5. Ethernet cable
6. Power adapter for nvidia jetson nano
7. 3D printer.(Not necessary)
8. A latop(Nvidia GPU preferred) or any cloud service provider.

# Software requirements(If running on Laptop)
1. Ubuntu machine(16.04 preferred).
2. Install anaconda.
3. Install the required dependencies. Some libraries like pytorch, opencv would require a little extra attention.
```conda env create -f deepWay.yml```
4. 

# Software Requirements(Jetson nano)
1. Follow [these](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) instructions for starting up with Jetson nano.
2. For connecting headless with jetson nano(using ethernet cable). <br>
```
ifconfig
Check inet addresss
nmap -sn inet_address/24 --> will return live ip address.
ssh machine_name@ip
Enter password
Now you can connect switch on desktop sharing
Now connect to jetson using Reminna.

```
3. Now install all the required dependicies. 



### 1. Collecting dataSet and Generating image masks.
I made videos of roads and converted those videos to jpg's. This way I collected a dataSet of approximately 10000 images.I collected images from left, right and center view. e.g:<br>
<img src="readMe/left.jpg" height=150  hspace=20px vspace=200px/>
<img src="readMe/center.jpg" height=150 hspace=20px vspace=20px/>
<img src="readMe/right.jpg" height=150 hspace=20px vspace=20px/><br>  
    
For Unet, I had to create binary masks for the input data, I used LabelBox for generating binary masks. (This took a looooooooot of time). A sample is as follows-><br> 
<img src="readMe/12.jpg" height=170 hspace=20px vspace=20px/>
<img src="readMe/12_mask.jpg" height=170 hspace=20px vspace=20px/><br>  
   
### 2. Model training
I trained a U-Net based model for road segmentation on Azure.
The loss(pink:traning, green:validation) vs iterations curve is as follows.<br>
<img src="readMe/loss.svg" height=400px/>
<br>
### 3. 3D modelling and printing
My friend Sangam Kumar Padhi helped me with CAD model. You can look at it [here](https://github.com/satinder147/DeepWay.v2/blob/master/3D%20model/model.STL)

### 4. Electronics on the spectacles




# This is the project

# People to Thank
1. Army Institute of Technology (My college)
2. Prof. Avinash Patil,Sangam Kumar Padhi, Sahil and Priyanshu for 3D modelling and printing
3. Shivam sharma and Arpit for data labelling
4. Nvidia for providing a free jetson kit
5. LabelBox: For providing me with the free license of their **Amazing Prodcut**.

# References
1. [Pytorch](https://pytorch.org/)
2. [PyimageSearch](https://www.pyimagesearch.com/)
3. [Pytorch Community](https://discuss.pytorch.org/)
4. [AWS](https://aws.amazon.com/)
5. [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
6. [U-Net implementation(usuyama)](https://github.com/usuyama/pytorch-unet)
7. [U-Net implementation(Heet Sankesara)](https://towardsdatascience.com/u-net-b229b32b4a71)

