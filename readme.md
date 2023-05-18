# Robot Reading

This is the repository concerning the code for my bachelor's thesis on the topic of "Robot Reading". It includes the main code of the pipeline, a helper script for calculating the text similarity, the images recorded on the PR2 robot and the results of the text recognition.

The main.py has four different modes:
- Full pipeline on the PR2 images
- Text detection only on the PR2 images
- Full Pipeline on a single image in the ```/camera``` directory
- Live version utilizing a webcam (Target text can be set)

Fixing cv2.imshow error for live demo:
```
pip install opencv-python==3.4.17.61
pip uninstall opencv-python-headless
pip uninstall opencv-python
pip install opencv-python
```

Otherwise, the images can also be displayed using PLT, though it doesn't automatically update the frame for the live version:
```
plt.imshow()
plt.show()
```