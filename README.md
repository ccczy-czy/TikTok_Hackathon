Its best to create a python env (venv) so downloading all this does not mess up any other python projects you have going on. 

pip3 install -r requirements.txt
to install all necessary dependencies. 



NOTES: 
If testing, i have added the actual loaded model files to the .gitignore file since they are too big for github. So for now you have to run the ImageEncoder.py and the TextEncoder.py to load them in you local machine every time. Later on we can load them from some sotrage system (s3?)