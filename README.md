# EE

This is an EE implementation using table-filling for Du_EE dataset


## Requirements
* Python (tested on 3.7.11)
* numpy (tested on 1.21.5)
* CUDA (tested on 11.3)
* PyTorch (tested on 1.10.1)
* Transformers (tested on 4.15.0)

## Files
bert-base-chinese is the bert pretrain model for chinese  
checkpoint saves the best model parameter  
datasets contains the Du_EE dataset  

data.py reads the Du_EE dataset and constructs the table  
model.py implements the table-filling model  
train.py and test.py are for train and test  
utils.py contains some data structure and functions  


## Train and Test

```bash
>> python train.py  # for train
>> python test.py  # for test
```