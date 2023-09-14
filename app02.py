import numpy as np
import gradio as gr
from tensorflow.keras import models

from recording_helper import record_audio
from tf_helper import preprocess_audiobuffer
commands =['down' ,'go', 'left', 'right', 'stop' ,'up']
loaded_model = models.load_model("saved")

def predict_mic(audio, state=""):
    
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]] 
    state += command + " "
    return state, state   
    

gr.Interface(
    fn=predict_mic,
    inputs=[
        gr.Audio(source="microphone", type="filepath", streaming=True),
        "state"
    ],
    outputs=[
        "textbox",
        "state"
    ],
    live=True).launch()