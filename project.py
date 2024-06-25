import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import simpleaudio as sa
import threading

def draw_distance_line(frame, original_dims, draw_line):
    original_height, original_width = original_dims
    height, width, _ = frame.shape

    scale_x = width / original_width
    scale_y = height / original_height

    yellow_trap_width = 380 * scale_x
    green_trap_width = 220 * scale_x

    # Współrzędne środka samochodu
    car_center = width // 2

    red_y_top = int(height * 0.8)
    red_y_bottom = height
    red_x_left_top = int((original_width // 2 - 450) * scale_x)
    red_x_right_top = int((original_width // 2 + 450) * scale_x)
    red_x_left_bottom = int((original_width // 2 - 600) * scale_x)
    red_x_right_bottom = int((original_width // 2 + 600) * scale_x)

    yellow_y_top = int(height * 0.6)
    yellow_y_bottom = int(height * 0.8)
    yellow_x_left_top = car_center - int(yellow_trap_width) + int(80 * scale_x)
    yellow_x_right_top = car_center + int(yellow_trap_width) - int(80 * scale_x)
    yellow_x_left_bottom = red_x_left_top
    yellow_x_right_bottom = red_x_right_top

    green_y_top = int(height * 0.5)
    green_y_bottom = int(height * 0.6)
    green_x_left_top = car_center - int(green_trap_width)
    green_x_right_top = car_center + int(green_trap_width)
    green_x_left_bottom = yellow_x_left_top
    green_x_right_bottom = yellow_x_right_top

    if draw_line:
        cv2.line(frame, (red_x_left_top, red_y_top), (red_x_left_bottom, red_y_bottom), (0, 0, 255), 2)
        cv2.line(frame, (red_x_left_top, red_y_top), (red_x_right_top, red_y_top), (0, 0, 255), 2)
        cv2.line(frame, (red_x_right_top, red_y_top), (red_x_right_bottom, red_y_bottom), (0, 0, 255), 2)
        cv2.line(frame, (red_x_right_bottom, red_y_bottom), (red_x_left_bottom, red_y_bottom), (0, 0, 255), 2)
        cv2.line(frame, (yellow_x_left_top, yellow_y_top), (yellow_x_left_bottom, yellow_y_bottom), (0, 255, 255), 2)
        cv2.line(frame, (yellow_x_left_top, yellow_y_top), (yellow_x_right_top, yellow_y_top), (0, 255, 255), 2)
        cv2.line(frame, (yellow_x_right_top, yellow_y_top), (yellow_x_right_bottom, yellow_y_bottom), (0, 255, 255), 2)
        cv2.line(frame, (yellow_x_right_bottom, yellow_y_bottom), (yellow_x_left_bottom, yellow_y_bottom), (0, 255, 255), 2)
        cv2.line(frame, (green_x_left_top, green_y_top), (green_x_left_bottom, green_y_bottom), (0, 255, 0), 2)
        cv2.line(frame, (green_x_left_top, green_y_top), (green_x_right_top, green_y_top), (0, 255, 0), 2)
        cv2.line(frame, (green_x_right_top, green_y_top), (green_x_right_bottom, green_y_bottom), (0, 255, 0), 2)
        cv2.line(frame, (green_x_right_bottom, green_y_bottom), (green_x_left_bottom, green_y_bottom), (0, 255, 0), 2)

    return (red_x_left_top, red_y_top, red_x_right_bottom, red_y_bottom)

def detect_objects(root, choices, draw_line, play_sound):
    classifiers = []
    for choice in choices:
        if choice == 'Piesi':
            classifiers.append(cv2.CascadeClassifier('pedestrian.xml'))
        elif choice == 'Samochody':
            classifiers.append(cv2.CascadeClassifier('cars.xml'))
        else:
            print("Nieprawidłowy wybór:", choice)

    file_path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    def play_alarm():
        wave_obj = sa.WaveObject.from_wave_file('alarm.wav')
        play_obj = wave_obj.play()
        play_obj.wait_done()

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_dims = (original_height, original_width)
    aspect_ratio = original_width / original_height

    max_width = 800
    max_height = 600

    if original_width > max_width or original_height > max_height:
        if aspect_ratio > 1:
            screen_width = max_width
            screen_height = int(max_width / aspect_ratio)
        else:
            screen_height = max_height
            screen_width = int(max_height * aspect_ratio)
    else:
        screen_width = original_width
        screen_height = original_height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (screen_width, screen_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        red_x_left_top, red_y_top, red_x_right_bottom, red_y_bottom = draw_distance_line(frame, original_dims, draw_line)

        alarm_played = False
        for classifier in classifiers:
            objects = classifier.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if (red_x_left_top <= x <= red_x_right_bottom or red_x_left_top <= x + w <= red_x_right_bottom) and (red_y_top <= y <= red_y_bottom or red_y_top <= y + h <= red_y_bottom):
                    if not alarm_played and play_sound:
                        threading.Thread(target=play_alarm).start()
                        alarm_played = True

        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', screen_width, screen_height)
        cv2.moveWindow('Video', (root.winfo_screenwidth() - screen_width) // 2, (root.winfo_screenheight() - screen_height) // 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Wybór detekcji obiektów")

    root.withdraw()
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('+{}+{}'.format((screen_width - root.winfo_reqwidth()) // 2, (screen_height - root.winfo_reqheight()) // 2))
    root.deiconify()

    def on_button_click():
        selected_choices = [choice.get() for choice in checkboxes if choice.get()]
        draw_line = draw_line_var.get()
        play_sound = play_sound_var.get()
        detect_objects(root, selected_choices, draw_line, play_sound)

    choices = ['Piesi', 'Samochody']
    checkboxes = []
    for choice in choices:
        var = tk.StringVar()
        checkbox = tk.Checkbutton(root, text=choice, variable=var, onvalue=choice, offvalue="")
        checkbox.pack()
        checkboxes.append(var)

    start_button = tk.Button(root, text="Start", command=on_button_click)
    start_button.pack()

    draw_line_var = tk.BooleanVar()
    draw_line_var.set(True)
    draw_line_checkbox = tk.Checkbutton(root, text="Rysuj linie", variable=draw_line_var)
    draw_line_checkbox.pack()

    play_sound_var = tk.BooleanVar()
    play_sound_var.set(True)
    play_sound_checkbox = tk.Checkbutton(root, text="Odtwarzaj sygnał dźwiękowy", variable=play_sound_var)
    play_sound_checkbox.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
