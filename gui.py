import tkinter as tk
from tkinter import Button, Canvas
import numpy as np
from PIL import ImageGrab, Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DigitApp:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Recogniser")

        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas_bg = "white"

        self.canvas = Canvas(
            master,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.canvas_bg,
        )
        self.canvas.pack()

        self.button_frame = tk.Frame(master)
        self.button_frame.pack()

        self.predict_button = Button(
            self.button_frame, text="Predict", command=self.predict_digit
        )
        self.predict_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_button = Button(
            self.button_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)

        # Load PyTorch model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(
            torch.load("model/mnist_cnn.pt", map_location=self.device)
        )
        self.model.eval()

    def draw(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    def predict_digit(self):
        # Grab the canvas content as image
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height

        image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")
        image = image.resize((28, 28))
        image_arr = np.array(image)

        # Invert and normalize pixels
        image_arr = 255 - image_arr
        image_arr = image_arr / 255.0
        image_arr = image_arr.astype(np.float32)
        image_arr = np.expand_dims(image_arr, axis=0)  # shape (1, 28, 28)
        image_arr = np.expand_dims(image_arr, axis=0)  # shape (1, 1, 28, 28)

        image_tensor = torch.from_numpy(image_arr).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = F.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)
            digit = pred.item()
            confidence = confidence.item()

        self.result_label.config(
            text=f"Prediction: {digit} (Confidence: {confidence:.2f})"
        )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="")


def main():
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
