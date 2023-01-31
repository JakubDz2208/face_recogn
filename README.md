# Face_recogn

Face recogn is program, that recognize your facial expression by 7 categories like:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprised

By using your webcam and scanning your face it in real time.

## Get started

To start using Face recogn you have to install Python 3.x, then using comamnd line install packages from requirements.txt

```pip install -r /path/to/requirements.txt```

Then move to project folder and just run ``` python main.py ```

## File list
In project there are 4 files like main.py, model.py, face_recogn.h5 and haarcascade_frontalface_default.xml
### main.py
  This is main file, which use our saved neural network model and implement it to our camera. The haarcascade_frontalface_default.xml is used for finding face and face_recogn.h5 for figure out and display our current expresion.
  
  
  Neutral face:
  
  
  ![image](https://user-images.githubusercontent.com/44981301/215794039-1dacbcf8-be7f-43f9-9921-287e9d046f6b.png)
  
  
  Happy face:
  
  
  ![image](https://user-images.githubusercontent.com/44981301/215794230-d767ca41-926b-4ced-a585-9bf0a353b44f.png)
  
  
  Surprised face:
  
  
  ![image](https://user-images.githubusercontent.com/44981301/215794299-d387ccc8-609d-4f71-9b20-4aa03d55532c.png)



### model.py
  Here we have our CNN (convolutional neural network) model, there are training and validation images to get best accuracy 
