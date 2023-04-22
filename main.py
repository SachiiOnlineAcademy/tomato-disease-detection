from tkinter import *
from tkinter import filedialog

import tensorflow as tf
from PIL import Image, ImageTk


class PlantDiseaseGUI:
    def __init__(self, master):
        self.image_label = None
        self.master = master
        self.master.title("Plant Disease Detection")
        self.master.geometry("800x600")
        self.master.config(background="#1E1E1E")

        # Set default position of window to center of screen
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_cordinate = int((screen_width / 2) - (800 / 2))
        y_cordinate = int((screen_height / 2) - (600 / 2))
        self.master.geometry("{}x{}+{}+{}".format(800, 600, x_cordinate, y_cordinate))

        self.diseases = ["Tomato_Bacterial___spot", "Tomato_Early___blight", "Tomato___healthy",
                         "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                         "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                         "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

        self.disease_label = Label(self.master, text="Diseases: " + ", ".join(self.diseases), font=("Arial", 16),
                                   fg="#FFFFFF", bg="#1E1E1E")
        self.disease_label.pack(pady=(20, 10))

        self.file_path = ""
        self.image_frame = Frame(self.master, width=256, height=256, bg="#3B3B3B")
        self.image_frame.pack(pady=10)

        self.result_label = Label(self.master, text="Results will be displayed here", font=("Arial", 14), fg="#FFFFFF",
                                  bg="#1E1E1E")
        self.result_label.pack(pady=10)

        self.open_button = Button(self.master, text="Open Image", font=("Arial", 12), command=self.open_image,
                                  bg="#4CAF50", fg="#FFFFFF", activebackground="#4CAF50", activeforeground="#FFFFFF")
        self.open_button.pack(pady=10)

        self.recommendation_label = Label(self.master, text="Recommendations", font=("Arial", 16), fg="#FFFFFF",
                                          bg="#1E1E1E")
        self.recommendation_label.pack(pady=10)

        self.recommendation_box = Text(self.master, font=("Arial", 12), fg="#FFFFFF", bg="#3B3B3B", height=10,
                                       state="normal")
        self.recommendation_box.insert(END, "Here are some recommendations:\n\n")
        self.recommendation_box.insert(END, "Recommendation 1\n")
        self.recommendation_box.insert(END, "Recommendation 2\n")
        self.recommendation_box.insert(END, "Recommendation 3\n")
        self.recommendation_box.pack(pady=(0, 20))
        self.recommendation_box.configure(state="disabled")
        # Create a tag for centering the text
        self.recommendation_box.tag_configure("center", justify="center")

        # Center the text in the recommendation box
        self.recommendation_box.tag_add("center", "1.0", "end")

    def open_image(self):
        self.clear_image()
        self.file_path = filedialog.askopenfilename()
        self.display_image()
        # Load the pre-trained model
        model = tf.keras.models.load_model('C:/Users/user/PycharmProjects/pythonModelGUI'
                                           '/newnew_trained_model.h5')
        # Preprocess the image
        img = tf.keras.preprocessing.image.load_img(self.file_path, target_size=(256, 256))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make prediction using the loaded model
        pred = model.predict(img_array)

        # Get the predicted label
        label = self.diseases[pred.argmax()]
        self.result_label.configure(text="The result is: " + label)
        # Update the recommendation box with the predicted label

        recommendations = {
            "Tomato_Bacterial___spot":
                "Recommendation 1 for Tomato_Bacterial___spot\nRecommendation 2 for "
                "Tomato_Bacterial___spot\nRecommendation 3 for"
                "Tomato_Bacterial___spot",
            "Tomato_Early___blight": "Recommendation 1 for Tomato_Early___blight\nRecommendation 2 for "
                                     "Tomato_Early___blight\nRecommendation 3 for Tomato_Early___blight",
            "Tomato___Late_blight": "Recommendation 1 for Tomato___Late_blight\nRecommendation 2 for "
                                    "Tomato___Late_blight\nRecommendation 3 for Tomato___Late_blight",
            "Tomato___healthy": "Recommendation 1 for Healthy plants\nRecommendation 2 for Healthy "
                                "plants\nRecommendation 3 for"
                                "Healthy plants",
            "Tomato___Leaf_Mold": "Recommendation 1 for Tomato___Leaf_Mold\nRecommendation 2 for Tomato___Leaf_Mold"
                                  "\nRecommendation 3 for Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot": "Recommendation 1 for Tomato___Septoria_leaf_spot\nRecommendation 2 for "
                                           "Tomato___Septoria_leaf_spot"
                                           "\nRecommendation 3 for Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite":"Tomato___Spider_mites "
                                                            "Two-spotted_spider_mite\nTomato___Spider_mites "
                                                            "Two-spotted_spider_mite\nTomato___Spider_mites "
                                                            "Two-spotted_spider_mite",
            "Tomato___Target_Spot":"Tomato___Target_Spot\nTomato___Target_Spot\nTomato___Target_Spot",
            "Tomato___Tomato_mosaic_virus":"Tomato___Tomato_mosaic_virus\nTomato___Tomato_mosaic_virus"
                                           "\nTomato___Tomato_mosaic_virus",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Tomato___Tomato_Yellow_Leaf_Curl_Virus"
                                                     "\nTomato___Tomato_Yellow_Leaf_Curl_Virus"
                                                     "\nTomato___Tomato_Yellow_Leaf_Curl_Virus"

        }
        self.recommendation_box.configure(state="normal")
        self.recommendation_box.delete(1.0, END)
        self.recommendation_box.insert(END, "Here are the recommendations:\n\n")
        self.recommendation_box.insert(END, recommendations.get(label))
        self.recommendation_box.configure(state="disabled")
        # Create a tag for centering the text
        self.recommendation_box.tag_configure("center", justify="center")

        # Center the text in the recommendation box
        self.recommendation_box.tag_add("center", "1.0", "end")

    def clear_image(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

    def display_image(self):
        if self.file_path:
            image = Image.open(self.file_path)
            image = image.resize((256, 256), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            self.image_label = Label(self.image_frame, image=photo, bg="#3B3B3B")
            self.image_label.image = photo
            self.image_label.pack(fill=BOTH, expand=YES)
        else:
            self.image_label.config(image=None)
            self.image_label.image = None


root = Tk()
PlantDiseaseGUI(root)
root.mainloop()
