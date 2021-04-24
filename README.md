# AI Project-Image Captioningin Noisy Dataset

To make problem hard, there are some noisy and wrong images and captions in original data. 
Therefore, I remove those of data and fix corresponded captions. You can download image data in [here](https://drive.google.com/drive/folders/1Xtk-EK0-CfwfNWShvjF8RpAnjZbBePMA?usp=sharing). 

If you want to train the model, you can use
```bash
$ python train.py    
```
Note that there are lots of parameters. So you can try your combination.

If you want to test the model and predict the given images, you can use
```bash
$ python test.py --ENCODER_MODEL_LOAD_PATH <encoder path> --DECODER_MODEL_LOAD_PATH <decoder_path>
```
Note that test images have to be in <test_img> folder as default setting. Or you can change the directory as follows:
```bash
$ python test.py --ENCODER_MODEL_LOAD_PATH <encoder path> --DECODER_MODEL_LOAD_PATH <decoder path> --test_img_dir <test img dir>
```

