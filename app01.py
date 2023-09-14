import numpy as np
from tensorflow.keras import models
from recording_helper import record_audio
from tf_helper import preprocess_audiobuffer

commands =['down' ,'go', 'left', 'right', 'stop' ,'up']

loaded_model = models.load_model("saved")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("predict_label:", command)
    return command

if __name__ == "__main__":
    
    while True:
        command = predict_mic()
        if command == "stop":
            break