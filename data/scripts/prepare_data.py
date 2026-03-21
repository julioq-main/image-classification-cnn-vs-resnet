#Python Script to build the dataset from other. We are dividing our data into test, val  and train

import splitfolders

#Set path to get data and to output it
input_path = "image-classification-cnn-vs-resnet/data/raw"
output_name = "image-classification-cnn-vs-resnet/data/processed"

#split all categories folders into train, val and test folders 

splitfolders.ratio(input_path, output=output_name,
                    seed=1234, ratio=(0.8,0.1,0.1), 
                    group_prefix=None, move=False)