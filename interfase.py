import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
import os
import trainingmodel as model
import speech_recognition as sr

# Función para entrenar el modelo desde la interfaz
def entrenar_modelo_interfaz():
    
    model.entrenar_modelo()
    messagebox.showinfo("Éxito", "Modelo entrenado correctamente")

def evaluar_modelo_interfaz():
    model_path = "my_model.h5"
    model.evaluar_modelo(model_path)
    messagebox.showinfo("Éxito", "Modelo evaluado correctamente")

#Cargar Imagen para probar el modelo entrenado
def cargar_imagenes():
    path_image = filedialog.askopenfilename(title="Seleccione la imagen")
    if not path_image:
        messagebox.showwarning("Advertencia", "Debe seleccionar una imagen")
        return
    
    # Predecir con el modelo entrenado
    predecir_con_modelo(path_image)

def predecir_con_modelo(path_image):
    model_path = 'my_model.h5'
    if not os.path.exists(model_path):
        messagebox.showwarning("Advertencia", "Debe entrenar el modelo antes de utilizarlo")
        return

    # Cargar el modelo entrenado
    model_path = 'my_model.h5'

    result = model.predecir_con_modelo_entrenado(model_path, path_image)
    if result:
        messagebox.showinfo("Resultado", "Se detectó glaucoma en la imagen")
    else:
        messagebox.showinfo("Resultado", "No se detectó glaucoma en la imagen")

def comando_voz():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di 'cargar imagen' para cargar una imagen.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio, language="es-ES")
            print("Comando reconocido:", command)
            if "cargar imagen" in command:
                cargar_imagenes()
            else:
                print("Comando no reconocido.")
        except sr.UnknownValueError:
            print("No se pudo entender el audio.")
        except sr.RequestError as e:
            print("Error al solicitar resultados del reconocimiento de voz; {0}".format(e))

#

# Crear la interfaz gráfica
root = tk.Tk()
root.title("Glaucoma Detection")
root.geometry("800x600")
root.configure(bg="lightblue")

fuente = ("Arial", 12)

# Establecer el icono de la aplicación
logo = tk.PhotoImage(file="images/logo.png")
root.iconphoto(True, logo)

# Crear un Label para la imagen de fondo
background_image = tk.PhotoImage(file="images/background.png")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Sección de entrenamiento y validación del modelo
frame_entrenamiento_validacion = tk.Frame(root, bg="lightblue")
frame_entrenamiento_validacion.pack(pady=(5), anchor='center', fill="both")

label_entrenamiento = tk.Label(frame_entrenamiento_validacion, text="Entrenamiento y Validación del Modelo", font=fuente, bg="lightblue")
label_entrenamiento.pack()

btn_entrenar = tk.Button(frame_entrenamiento_validacion, text="Entrenar Modelo", font=fuente, bg="white", command=entrenar_modelo_interfaz)
btn_entrenar.pack(fill="both")

btn_evaluar = tk.Button(frame_entrenamiento_validacion, text="Validar Modelo", font=fuente, bg="white", command=evaluar_modelo_interfaz)
btn_evaluar.pack(fill="both")


# Sección de carga de imágenes y comando de voz
frame_imagenes = tk.Frame(root, bg="lightblue")
frame_imagenes.pack(pady=10, anchor='center', fill="both")  # Alineación hacia el centro

label_imagenes = tk.Label(frame_imagenes, text="Carga de Imágenes y Comando de Voz", font=fuente, bg="lightblue")
label_imagenes.pack()

# Botón para cargar imágenes
btn_cargar_imagenes = tk.Button(frame_imagenes, text="Cargar Imagen", font=fuente, bg="white", command=cargar_imagenes)
btn_cargar_imagenes.pack(fill="both")

# Botón para el comando de voz
btn_comando_voz = tk.Button(frame_imagenes, text="Comando de Voz: Haz Click y Di 'cargar imagen' para cargar una imagen.", font=fuente, bg="white", command=comando_voz)
btn_comando_voz.pack(fill="both")

root.mainloop()
