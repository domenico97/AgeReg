import sys
import os

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

# Dizionario delle etnie
ethnicitiesDict = {'Occidentale': 0, 'Africana': 1, 'Asiatica Orientale': 2, 'Asiatica centro - meridionale': 3, 'Non classificato': 4}
# Lista delle descrizioni delle etnie
ethnicitiesDescriptions = list(ethnicitiesDict.keys())
# Dizionario dei modelli
modelsDict = {'Img/Sex Colori': 'RGBsex_200_0.15_3CL_0.2.h5', 'Img Colori': 'RGBimg_200_0.15_3CL_0.2.h5', 'Img/Sex/Eth Colori': 'RGBall_200_0.15_3CL_0.2.h5', 'Img Grigi': 'img_200_0.15_3CL_0.2.h5'}
# Lista delle descrizioni dei modelli
modelsDescriptions = list(modelsDict.keys())


# Tipo del modello da caricare: 0 = solo immagini, 1 = immagini e sesso, 2 = tutte le features
modelType = 0
# Numero di canali dell'immagine da caricare. Default = 1 (Scala di grigi)
imageChannels = 1
# Tipo di conversione di colori da effettuare. Default = Conversione in scala di grigi
colorConversion = cv2.COLOR_BGR2GRAY
# Sesso dell'utente
sex = 0
# Etnia dell'utente
ethnicity = 0


def main():
    # Lista degli age_buckets da predire
    AGE_BUCKETS = ["(0-9)", "(10-19)", "(20-29)", "(30-39)", "(40-49)", "(50-59)", "(60-69)", "(70-79)", "(80-89)", "(90-99)", "(100+)"]
    # Inizializzazioni delle variabili per effettuare la media dell'età
    mean, sum, n = 0, 0, 0

    # Tema della GUI
    sg.ChangeLookAndFeel('DefaultNoMoreNagging')

    # Caricamento del modello di face detection
    print("[INFO] loading face detector model...")
    faceNet = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Caricamento del modello di age detection
    print("[INFO] loading age detector model...")
    ageNet = tf.keras.models.load_model("Modelli/img_200_0.15_3CL_0.2.h5")

    # Layout della finestra principale
    layout = [[sg.Image(filename='', key='image')],
              [sg.Button('Exit', font='Any 1', image_filename='Immagini/exit.png', image_subsample=9, button_color=('#F0F0F0', sg.theme_background_color()), border_width=0),
               sg.Button('Settings', font='Any 1', image_filename='Immagini/settings.png', image_subsample=9, button_color=('#F0F0F0', sg.theme_background_color()), border_width=0)]]

    # Creazione della finestra principale
    main_window = sg.Window('AgeReg', use_default_focus=False)
    main_window.Layout(layout).Finalize()

    # Definizione sorgente di input (webcam)
    cap = cv2.VideoCapture(0)

    # loop che legge e mostra i frame acquisiti dalla webcam
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

                if settings_button == 'Submit':
                    print("[INFO] Settings button was pressed.")
                    ageNet = setModel(modelsDict[setting_values['modelToUse']])
                    #print(setting_values)
                    global sex
                    sex = 0 if setting_values['Uomo'] else 1
                    global ethnicity
                    ethnicity = ethnicitiesDict[setting_values['ethnicity']]
                    #print(setting_values['modelToUse'], sex, ethnicity)
                    settings_window.close()
                    break

                if settings_button == 'Cancel':
                    print("[INFO] Cancel button was pressed.")
                    settings_window.close()
                    break

        # Lettura frame di input
        ret, frame = cap.read()

        # Detection dei volti e per ognuno predizione dell'età
        results = detect_and_predict_age(frame, faceNet, ageNet)

        # Se non viene rilevato nessun volto o ne vengono rilevati più di uno vengono resettati i contatori
        # In questi casi non è possibile fare la media delle età
        if not results or len(results) > 1:
            sum, mean, n = 0, 0, 0

        if len(results) == 1:
            # Incremento del contatore del numero di frame letti fin dall'inizio
            n += 1
            # Estrazione dell'età predetta
            age = results[0]["age"][0][0]
            # Media delle età
            sum += age
            mean = sum / n

            #text = "{}  {}".format(mean.astype(int), age.astype(int))
            text = "{}".format(mean.astype(int))

        for r in results:

            # Se viene rilevato più di un volto si utilizza la classificazione con gli age_buckets
            if len(results) > 1:
                 # Estrazione dell'età predetta
                 age = r["age"][0][0].astype(int)
                 # Posizionamento dell'età nell'age_bucket appropriato
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
            # Disegno degli angoli della face detection
            cv2.line(frame, (startX, startY), (startX + 50, startY), (255, 255, 255), 2)
            cv2.line(frame, (startX, startY), (startX, startY + 50), (255, 255, 255), 2)

            cv2.line(frame, (endX, startY), (endX - 50, startY), (255, 255, 255), 2)
            cv2.line(frame, (endX, startY), (endX, startY + 50), (255, 255, 255), 2)

            cv2.line(frame, (startX, endY), (startX + 50, endY), (255, 255, 255), 2)
            cv2.line(frame, (startX, endY), (startX, endY - 50), (255, 255, 255), 2)

            cv2.line(frame, (endX, endY), (endX - 50, endY), (255, 255, 255), 2)
            cv2.line(frame, (endX, endY), (endX, endY - 50), (255, 255, 255), 2)
            # Disegno dell'età in posizione centrale
            middleX = int((startX + endX)/2)
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 8)[0]
            middleX_age = int(middleX - (textsize[0]/2))
            cv2.putText(frame, text, (middleX_age, startY),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 8)

            textsize = cv2.getTextSize("anni", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            middleX_str = int(middleX - (textsize[0]/2))
            cv2.putText(frame, "anni", (middleX_str, startY + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        main_window.FindElement('image').update(data=imgbytes)


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

    # Inizializzazione della lista dei risultati
    results = []
    # Detection dei volti
    faces = faceNet.detectMultiScale(frame, 1.1, 20)

    # Estraggo le coordinate del volto
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w, :]
        #print(frame.shape)
        #print(face.shape)

        # Resize del volto per il passaggio al modello
        face = cv2.resize(face, dsize=(80, 80))
        #cv2.imshow("avc", face)
        #key = cv2.waitKey(1)
        #face = color.rgb2gray(face)

        # Conversione di colore del volto
        face = cv2.cvtColor(face, colorConversion)
        face = face / 255
        face = face.reshape(-1, face.shape[0], face.shape[1], imageChannels)
        #face = np.reshape(face, (80,80,1))

        # Costruzione delle feature da passare al modello
        feat = None
        if modelType == 1:
            feat = np.array([sex])
        elif modelType == 2:
            feat = np.array([sex, ethnicity])


        #print(face.shape)
        #img = np.squeeze(face)
        #cv2.imshow("avc", img)
        #key = cv2.waitKey(1)

        # Predict dell'età dell'utente
        age = ageNet.predict(face) if modelType==0 else ageNet.predict([face,feat])
        endX = x + w
        endY = y + h

        # Costruzione di un dizionario contenente la bounding box del volto e l'età
        d = {
            "loc": (x, y, endX, endY),
            "age": age
            }
        # Update della lista
        results.append(d)
    return results


def createSettingsWindow():
    sex_layout = [
        [sg.Radio('Uomo', "SEX", key='Uomo', default=True), sg.Radio('Donna', "SEX", key='Donna')]
    ]

    model_layout = [
        [sg.Combo(values=modelsDescriptions, readonly=True, default_value='Img Grigi', size=(30, 3), key='modelToUse', change_submits=True, pad=(10, 10)),
         sg.Text(' ' * 10,),
         sg.Text('Configurazione:\n   Percentuale di Dropout : 0.15\n   # livelli convoluzionali : 3\n   % test set : 20%', key='explainer', pad=(10, 10))]
    ]
    ethnicity_layout = [
        [sg.Combo(values=ethnicitiesDescriptions, readonly=True, default_value=ethnicitiesDescriptions[0], size=(30, 3), key='ethnicity', change_submits=True, pad=(10, 10))]
    ]

    layout_setting = [
        [sg.Text(' '*25), sg.Text('Impostazioni', font=("Helvetica", 15)), sg.Text(' '*25), sg.Image(filename='Immagini/settings1.png', size=(50, 50))],
        [sg.Frame('Seleziona il modello da utilizzare', model_layout, font='Any 11', title_color='black', pad=(10, 20))],
        [sg.Text('_' * 60)],
        [sg.Frame('Sesso', sex_layout, font='Any 11', title_color='black', pad=(10,10)), sg.Frame('Etnia', ethnicity_layout, font='Any 11', title_color='black', pad=(10, 10))],
        [sg.Text('NOTA: Le scelte di etnia e sesso non verrano considerate in caso di \nriconoscimento contemporaneo di più individui')],
        [sg.Text('_' * 60)],
        [sg.Button('Cancel', font='Any 1', image_filename='Immagini/x.png', image_subsample=9, button_color=('White', sg.theme_background_color()), border_width=0),
         sg.Button('Submit', font='Any 1', image_filename='Immagini/ok.png', image_subsample=9, button_color=('White', sg.theme_background_color()), border_width=0)]]
    # Creazione della finestra di settings
    window_setting = sg.Window('Impostazioni', use_default_focus=False)
    window_setting.Layout(layout_setting).Finalize()

    return window_setting


def setModel(model):

    # Riferimento alle variabili globali
    global colorConversion
    global imageChannels
    global modelType

    # Setting delle configurazioni appropriate dei modelli
    if 'RGB' in model:
        colorConversion = cv2.COLOR_BGR2RGB
        imageChannels = 3
    else:
        colorConversion = cv2.COLOR_BGR2GRAY
        imageChannels = 1

    if 'img' in model:
        modelType = 0
    elif 'sex' in model:
        modelType = 1
    elif 'all' in model:
        modelType = 2

    # Caricamento del modello di face detection
    print("[INFO] loading age detector model:" + model)
    ageNet = tf.keras.models.load_model("Modelli/" + model)

    return ageNet


main()
