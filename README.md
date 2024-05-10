# CS791_IDS

## Dataset
Synthetically Generated Data : https://drive.google.com/file/d/1AgL6xkahmx19BFhQ0r8ESmfVuMyvWOFA/view?usp=sharing <br>
InBreast Dataset : https://drive.google.com/file/d/1knX6qcx3tEXauYQAdV_Da3ew0VxSa4_g/view?usp=sharing 

The test set contains augmented images. Remember to remove them before using

## Synthetic Data Generation
To generate images please go to the Pro-GAN folder and run the train.py file. You can modify the hyper-parameters by changing the config.py file. 

## Model Training (1st Step)
Run all the cells in autoencoder-reconstruction notebook. It will train the model and save the best performing models weights. 

## Model Training (2nd Step)
Run all the cells in autoencoder-classification notebook. It will train the model based on the proposed approach. 

Additionally, the autoencoder-classification model can be found in model.py file. main_DNN.ipynb contains codes to run the off-the-self DNN models for comparison.


