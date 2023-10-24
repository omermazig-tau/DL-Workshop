# NBA Shot Classifier Demo #

### Getting a dataset ###

You can run the [create_videos_bank.ipynb](create_videos_bank.ipynb) notebook to create it yourself, but there might be some dependency management required for that. For example, py-tesseract needs you to first manually install [the program manually](https://github.com/tesseract-ocr/tesseract#installing-tesseract)

If you don't want to deal with that, you can download a [big dataset (21 GB)](https://drive.google.com/drive/folders/1g7JBMC1XJAaqHzYRpsl5XhDaja-ZS7Vr?usp=sharing) or a [small dataset (256 MB)](https://drive.google.com/drive/folders/1ylXW32poiBJWdrjd9eGmEWvoDTRxyFlc?usp=sharing) from my drive, rename it `dataset`, and put it in the project's root folder.

### Checking dataset metrics ###

Once you have a dataset, just run the [check_model_inference.ipynb](check_model_inference.ipynb) notebook, and you will get the full classification report. 
This could take a while, so you might want to start with a small dataset. 

## Predicting a single video ##

You can take any single video from the dataset, and upload it to the 
[dedicated space on huggingface](https://huggingface.co/spaces/omermazig/videomae-finetuned-nba-5-class).
You should get a prediction within about 30 seconds.
There's also a [space](https://huggingface.co/spaces/omermazig/videomae-finetuned-nba-5-class-multilabel) for multi label prediction, but as stated in `Project.pdf`, it is not as accurate.  

This spaces should also support downloaded videos from YouTube 
(Tested it on a few and got good results), so you can also try that.