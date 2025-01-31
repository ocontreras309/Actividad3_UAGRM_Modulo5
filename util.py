from threading import Thread
from PIL import Image, ImageTk

import psutil
import tkinter as tk

def show_chessboard(queens):
    image_window = tk.Tk()
    board_image = Image.open('8x8_board.png')
    queen_image = Image.open('queen.png')

    for x, y in enumerate(queens):
        board_image.paste(queen_image, (int(x * 75), int(y * 75)), mask=queen_image)
    
    img = ImageTk.PhotoImage(board_image)
    background = tk.Label(image_window, image=img)
    background.pack(side="bottom", fill="both", expand="yes")

    image_window.mainloop()

class MonitoringThread(Thread):
    def __init__(self):
        super().__init__()
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        print('CPU usage,Mem usage')

        while self.running:
            print(str(psutil.cpu_percent(interval=0.1)) + "," + str(psutil.virtual_memory().percent))
