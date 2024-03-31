import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd

# Trainiertes Modell laden
model = load_model('C:/Users/elesh/Desktop/University/PEiM/my_model.h5')

# Klassennamen basierend auf dem Training des Modells definieren
class_names = ['neuwertig', 'mittel', 'defekt']

# Titel der Streamlit-App
st.title('App zur Erkennung von Werkzeugverschleiß')

# Datei-Uploader, damit der Benutzer Bilder hochladen kann
uploaded_file = st.file_uploader("Laden Sie ein Bild des Werkzeugs hoch", type=["jpg", "png"])
if uploaded_file is not None:
    # Hochgeladenes Bild anzeigen
    st.image(uploaded_file, caption='Hochgeladenes Werkzeugbild', use_column_width=True)
    
    # Hochgeladenes Bild vorverarbeiten
    def preprocess_image(uploaded_file):
        image = load_img(uploaded_file, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0  # Normalisieren auf [0,1]
        return image

    image = preprocess_image(uploaded_file)

    # Vorhersage treffen
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    
    # Vorhersage anzeigen
    st.write(f'Vorhersage: {predicted_class_name}')

# Formular für zusätzliche Daten
with st.form("tool_data"):
    st.write("Werkzeug- und Bearbeitungsinformationen")
    werkzeugtyp = st.text_input("Werkzeugtyp")
    vorschub = st.number_input("Vorschub (mm/Umdrehung)")
    drehzahl = st.number_input("Drehzahl (U/min)")
    zustellung = st.number_input("Zustellung (mm)")
    bauteilname = st.text_input("Bauteilname")
    bearbeitungsdauer = st.number_input("Bearbeitungsdauer pro Bauteil (Minuten)", step=1)

    # Schaltfläche zum Absenden des Formulars
    submitted = st.form_submit_button("Einreichen")
    if submitted:
        # Datensammlung
        werkzeugdaten = {
            "Werkzeugtyp": werkzeugtyp,
            "Vorschub": vorschub,
            "Drehzahl": drehzahl,
            "Zustellung": zustellung,
            "Bauteilname": bauteilname,
            "Bearbeitungsdauer": bearbeitungsdauer,
            "Vorhergesagter Zustand": predicted_class_name
        }
        
        # In DataFrame umwandeln
        df = pd.DataFrame([werkzeugdaten])
        st.write(df)
        
        # In CSV-Datei speichern
        df.to_csv("werkzeugverschleiss_daten.csv", mode='a', header=not st.file_exists("werkzeugverschleiss_daten.csv"), index=False)

        st.success("Daten erfolgreich gespeichert.")
