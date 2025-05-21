# How to install datasets

Download the different datasets and modify the corresponding config files in `configs` to put the path to the dataset in `dataset_dir` 

Datasets list:
- [How to install datasets](#how-to-install-datasets)
    - [Cityscapes](#cityscapes)
    - [COCO](#coco)
    - [PPDLS](#ppdls)

The instructions to prepare each dataset are detailed below. 

### Cityscapes
- Download the dataset from the [official website](https://www.cityscapes-dataset.com/) (you need to create an account for that)
- The directory structure should look like
```
cityscapes/
|–– leftImg8bit/
|   |–– train/ 
|   |–– val/
|   |–– test/
|–– gtFine/
|   |–– train/ 
|   |–– val/
|   |–– test/
```


### COCO
- Download the dataset from the [official website](https://cocodataset.org/#download) or using Fiftyone as explained in the official website page
- The directory structure should look like
```
coco/
|–– train2017/
|–– val2017/
|–– test2017/
|–– annotations
|   |–– instances_train2017.json
|   |–– instances_val2017.json

```


### PPDLS
- Download the dataset from the [official website](https://www.plant-phenotyping.org/datasets-download) (before downloading the dataset you should fill in a form and agree to the terms and conditions)
- The directory structure should look like (split.pkl is the random split train/val we fixed for the dataset, it is available in our repo)
```
CVPPP2017_LCC_training/
|–– split.pkl
|–– CVPPP2017_LCC_training/
|   |–– training
|       |–– A1
|       |–– A2
|       |–– A3
|       |–– A4

```