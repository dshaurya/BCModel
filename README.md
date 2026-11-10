# Instructions

You will need to install [python](https://www.python.org/downloads/) to run the model.  
Once you have downloaded the BC_Detection folder from drive(if needed unzip)  
**DO NOT UNZIP THE classifier.pt and detectorbc.pt files.**

Run the below in your computer's terminal/command prompt.

pip install ultralytics torch torchvision opencv-python pillow numpy pandas matplotlib gradio

Then just click on integrated_gui_bc.py to run it or enter this in your terminal

python integrated_gui_bc.py  
<br/>Then in your browser(preferable NOT Brave)

Enter this

<http://127.0.0.1:7860>

Then you can use the testing data folder to upload the images into the website and try out the model !

Currently classification of the cancer cells is not that good as seen in the evaluation folder with the metrics while detection is strong.(metrics in evaluation_output folder)
