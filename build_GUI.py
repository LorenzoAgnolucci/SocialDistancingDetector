import tkinter as tk
from VideoGet import VideoGet
import streamlink

def build_GUI():

    OptionList = [
        "Laptop webcam",
        "IP webcam",
        "Local video",
        "Link to stream"
    ]

    app = tk.Tk()
    app.title("Social Distancing Detector")
    app.geometry('750x500')

    optionVar = tk.StringVar(app)
    optionVar.set(OptionList[0])

    optionLabel = tk.Label(text="Select input video stream: ", font=20)
    optionLabel.place(x=10, y=10)

    opt = tk.OptionMenu(app, optionVar, *OptionList)
    opt.config(width=20, font=15)
    opt.place(x=250, y=5)

    sourceLabel = tk.Label(text="Don't worry about this", font=20)
    sourceLabel.place(x=10, y=60)

    sourceVar = tk.StringVar()
    sourceVar.set("0")

    sourceEntry = tk.Entry(app, width=50, textvariable=sourceVar, state="disabled")
    sourceEntry.place(x=10, y=90)

    calibrationLabel = tk.Label(text="Insert calibration matrix path (leave empty for no calibration)", font=20)
    calibrationLabel.place(x=10, y=150)

    calibrationVar = tk.StringVar()
    calibrationVar.set("calibration_matrix.yml")

    calibrationEntry = tk.Entry(app, width=50, textvariable=calibrationVar)
    calibrationEntry.place(x=10, y=180)

    def continue_callback():
        app.destroy()

    continueButton = tk.Button(app, text="Continue", font=20, fg="blue", command=continue_callback)
    continueButton.place(x=620, y=450)

    def option_callback(*args):
        sourceEntry.configure(state="normal")
        selectedOption = optionVar.get()

        if selectedOption == "Laptop webcam":
            sourceLabel.configure(text="Don't worry about this")
            sourceVar.set("0")
            sourceEntry.configure(state="disabled")
            calibrationVar.set("calibration_matrix.yml")

        elif selectedOption == "IP webcam":
            sourceLabel.configure(text="Insert the webcam IP")
            sourceVar.set("https://192.168.1.37:8080/video")
            calibrationVar.set("new_calibration_matrix.yml")

        elif selectedOption == "Local video":
            sourceLabel.configure(text="Insert the path to the local video")
            sourceVar.set("video/pedestrians.mp4")
            calibrationVar.set("")

        elif selectedOption == "Link to stream":
            sourceLabel.configure(text="Insert the link to the stream")
            sourceVar.set("https://www.youtube.com/watch?v=srlpC5tmhYs")
            calibrationVar.set("")

    optionVar.trace("w", option_callback)
    app.mainloop()

    final_selected_option = optionVar.get()
    calibration_path = calibrationVar.get()
    stream_source = sourceVar.get()
    video_getter = ""

    if final_selected_option == "Laptop webcam":
        stream_source = int(stream_source)
        video_getter = VideoGet(stream_source, True)

    elif final_selected_option == "IP webcam":
        video_getter = VideoGet(stream_source, True)

    elif final_selected_option == "Local video":
        video_getter = VideoGet(stream_source, False)

    elif final_selected_option == "Link to stream":
        stream_source = streamlink.streams(stream_source)['best'].url
        video_getter = VideoGet(stream_source, False)

    return video_getter, calibration_path


if __name__ == '__main__':
    build_GUI()
