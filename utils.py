import glob

# get the path/directory
folder_dir = 'assets'
 
# iterate over files in
# that directory
for images in glob.iglob(f'{folder_dir}/*'):
    # check if the image ends with png
    if (images.endswith((".jpeg","png"))):
        print(images)


