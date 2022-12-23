# PhoMoNet: Monocular Depth Estimation Network with Single-Pixel Depth Guidance
Official implementation of [Monocular Depth Estimation Network with Single-Pixel Depth Guidance](https://doi.org/10.1364/OL.478375)

## Results

##### visualization results:
<img src="https://github.com/jimmy9704/PhoMoNet/blob/main/image/Result.png" width="800"/>

## Web demo
[Here](https://7f46-163-152-183-111.jp.ngrok.io/) you can simulate simple SPAD guidance and visualize the results.

## Datasets
**NYU-Depth-v2**: Please follow the instructions in [BTS](https://github.com/cleinc/bts) to download the training/test set.

**RGB-SPAD**: Please follow the instructions in [single spad depth](https://github.com/computational-imaging/single_spad_depth) to download the test set.

## Testing
Pretrained models can be downloaded form [here](https://www.dropbox.com/s/tswsg84ga76yq9x/PhoMoNet_adabins.pt?dl=0).

To reproduce the reported results in our paper, follow these steps:
```
Step1: download the trained models and put it in the ./trained_models.
Step2: change the data and model paths in args_test_nyu.txt and args_test_real.txt
Step3: run "python evaluate.py args_test_nyu.txt" for NYU-Depth-v2 dataset
       run "python evaluate_real.py args_test_real.txt" for real RGB-SPAD dataset
```

## Acknowledgments
The code is based on [Adabins](https://github.com/shariqfarooq123/AdaBins) and [single spad depth](https://github.com/computational-imaging/single_spad_depth).

## Related works
* [SSD (single spad depth)](https://github.com/computational-imaging/single_spad_depth)
* [Adabins](https://github.com/shariqfarooq123/AdaBins)
* [BTS](https://github.com/cleinc/bts)
