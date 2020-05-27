import sys

if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf
from PIL import Image
import io
from sys import exit as exit

def main():
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-9)", "(10-19)", "(20-29)", "(30-39)", "(40-49)", "(50-59)", "(60-69)", "(70-79)","(80-89)","(90-99)","(100+)"]

    mean,sum,n = 0,0,0
    sg.ChangeLookAndFeel('DefaultNoMoreNagging')

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    faceNet = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # load our serialized age detector model from disk
    print("[INFO] loading age detector model...")
    ageNet = tf.keras.models.load_model("img_128_80_0.10_80.h5")

    # define the window layout
    layout = [[sg.Image(filename='', key='image')],
              [sg.Button('Exit',font='Any 1',image_filename='exit.png',image_subsample=9, button_color=('#F0F0F0',sg.theme_background_color()), border_width=0),
               sg.Button('Settings',font='Any 1',image_filename='settings.png',image_subsample=9, button_color=('#F0F0F0',sg.theme_background_color()), border_width=0)]]

    # create the window and show it without the plot
    main_window = sg.Window('AgeReg',use_default_focus=False)
    main_window.Layout(layout).Finalize()

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    #cap = cv2.VideoCapture(0)
    while True:
        button, values = main_window._ReadNonBlocking()

        if button is 'Exit' or values is None:
            print("[INFO] Exit button was pressed. Closing the program.")
            sys.exit(0)
        elif button == 'Settings':
            print("[INFO] Settings button was pressed.")
            settings_window = createSettingsWindow()
            while True:
                settings_button, setting_values = settings_window._ReadNonBlocking()
                if setting_values['modelToUse'] == 'Modello 1':
                    settings_window.FindElement('explainer').Update("Model 1")
                if setting_values['modelToUse'] == 'Modello 2':
                    settings_window.FindElement('explainer').Update("Model 2")
                if settings_button == 'Submit':
                    print("[INFO] Settings button was pressed.")
                    settings_window.close()
                    break
                if settings_button == 'Cancel':
                    print("[INFO] Cancel button was pressed.")
                    settings_window.close()
                    break



'''
        ret, frame = cap.read()

        # detect faces in the frame, and for each face in the frame,
        # predict the age
        results = detect_and_predict_age(frame, faceNet, ageNet)

        #Controllare se lista è vuota o ha più di un elemento per resettare i contatori
        if not results or len(results) > 1:
            sum, mean, n = 0, 0, 0

        if len(results) == 1:
            # contatore
            n += 1
            # predicted age
            age = results[0]["age"][0][0]
            sum += age
            mean = sum / n
            #text = "{}  {}".format(mean.astype(int), age.astype(int))
            text = "{}".format(mean.astype(int))
            # loop over the results

        for r in results:

            if len(results) > 1:
                 # predicted age
                 age = r["age"][0][0].astype(int)
                 digits = len(str(age))
                 if digits == 1:
                    age = AGE_BUCKETS[0]
                 elif digits == 3:
                    age = AGE_BUCKETS[10]
                 else:
                    i = str(age)
                    age = AGE_BUCKETS[int(i[0])]

                 text = "{}".format(age)

            (startX, startY, endX, endY) = r["loc"]
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.line(frame, (startX, startY), (startX + 50, startY), (255, 255, 255), 2)
            cv2.line(frame, (startX, startY), (startX, startY + 50), (255, 255, 255), 2)

            cv2.line(frame, (endX, startY), (endX - 50, startY), (255, 255, 255), 2)
            cv2.line(frame, (endX, startY), (endX, startY + 50), (255, 255, 255), 2)

            cv2.line(frame, (startX, endY), (startX + 50, endY), (255, 255, 255), 2)
            cv2.line(frame, (startX, endY), (startX, endY - 50), (255, 255, 255), 2)

            cv2.line(frame, (endX, endY), (endX - 50, endY), (255, 255, 255), 2)
            cv2.line(frame, (endX, endY), (endX, endY - 50), (255, 255, 255), 2)

            middleX = int((startX + endX)/2)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 8)[0]
            middleX_age = int(middleX - (textsize[0]/2))
            cv2.putText(frame, text, (middleX_age, startY),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)

            textsize = cv2.getTextSize("anni", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            middleX_str = int(middleX - (textsize[0]/2))
            cv2.putText(frame, "anni", (middleX_str, startY + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window.FindElement('image').update(data=imgbytes)
'''

'''def get_age(age):
    age = age
    if age >= 0 and age <= 3:return "0-3"
    if age >= 4 and age <= 10:return "4-10"
    if age >= 11 and age <= 20:return "11-20"
    if age >= 21 and age <= 30:return "21-30"
    if age >= 31 and age <= 40: return "31-40"
    if age >= 41 and age <= 50: return "41-50"
    if age >= 51 and age <= 60: return "51-60"
    if age >= 61 and age <= 70: return "61-70"
    if age >= 71 and age <= 80: return "71-80"
    if age >= 81 and age <= 90: return "81-90"
    if age >= 91: return "91-100+"'''


def detect_and_predict_age(frame, faceNet, ageNet):

    # initialize our results list
    results = []

    faces = faceNet.detectMultiScale(frame, 1.1, 20)
    cont = 0
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w, :]
        #print(frame.shape)
        #print(face.shape)
        cont = cont + 1

        face = cv2.resize(face, dsize=(80, 80))
        #cv2.imshow("avc", face)
        #key = cv2.waitKey(1)
        #face = color.rgb2gray(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face / 255
        face = face.reshape(-1, face.shape[0], face.shape[1], 1)
        #face = np.reshape(face, (80,80,1))

        #print(face.shape)
        #img = np.squeeze(face)
        #cv2.imshow("avc", img)
        #key = cv2.waitKey(1)

        age = ageNet.predict(face)
        endX = x + w
        endY = y + h
        # construct a dictionary consisting of both the face
        # bounding box location along with the age prediction,
        # then update our results list
        d = {
            "loc": (x, y, endX, endY),
            "age": (age)
            }
        results.append(d)

    # return our results to the calling function
    return results

def createSettingsWindow():
    sex_layout = [
        [sg.Radio('Uomo', "RADIO1", default=True), sg.Radio('Donna', "RADIO1")],
    ]

    # Layout Settings page
    layout_setting = [
        [sg.Text(' '*50),sg.Text('Impostazioni', font=("Helvetica", 13)),sg.Text(' '*50),sg.Image(filename='settings1.png',size=(50,50))],
        #[sg.Text('_' * 100, size=(70, 1))],
        [sg.Text('Modello da utilizzare', size=(70, 1))],
        [sg.InputCombo(['Modello 1', 'Modello 2'], size=(20, 3),key='modelToUse', change_submits=True), sg.Text('Descrizione Modello', key='explainer')],
        [sg.Text('_' * 100, size=(70, 1))],
        [sg.Frame('Sesso', sex_layout, font='Any 12', title_color='black')],
        [sg.Text('_' * 100, size=(70, 1))],
        [sg.Button('Cancel', button_color=('White','Red'), size=(10, 1), font='Any 14'),
         sg.Button('Submit', button_color=('White','Green'), size=(10, 1), font='Any 14')]]

    window_setting = sg.Window('Impostazioni')
    window_setting.Layout(layout_setting).Finalize()
    return window_setting
main()
