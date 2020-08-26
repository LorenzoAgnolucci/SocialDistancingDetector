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
    app.geometry('575x480')

    optionVar = tk.StringVar()
    optionVar.set(OptionList[0])

    optionFrame = tk.LabelFrame(app, text="Stream type", width=550, height=80)
    optionFrame.place(x=10, y=10)
    optionFrame.pack_propagate(False)

    optionLabel = tk.Label(optionFrame, text="Select input video stream: ", font=20)
    optionLabel.place(x=15, y=13)

    opt = tk.OptionMenu(optionFrame, optionVar, *OptionList)
    opt.config(width=20, font=15)
    opt.place(x=255, y=9)

    sourceFrame = tk.LabelFrame(app, text="Source", width=550, height=140)
    sourceFrame.place(x=10, y=110)
    sourceFrame.pack_propagate(False)

    sourceLabel = tk.Label(sourceFrame, text="Don't worry about this", font=20)
    sourceLabel.place(x=15, y=15)

    sourceVar = tk.StringVar()
    sourceVar.set("0")

    sourceEntry = tk.Entry(sourceFrame, width=55, textvariable=sourceVar, state="disabled")
    sourceEntry.place(x=15, y=65)

    calibrationFrame = tk.LabelFrame(app, text="Calibration", width=550, height=140)
    calibrationFrame.place(x=10, y=270)
    calibrationFrame.pack_propagate(False)

    calibrationLabel = tk.Label(calibrationFrame, text="Insert calibration matrix path (leave empty for no calibration)", font=20)
    calibrationLabel.place(x=15, y=15)

    calibrationVar = tk.StringVar()
    calibrationVar.set("webcam_calibration_matrix.yml")

    calibrationEntry = tk.Entry(calibrationFrame, width=50, textvariable=calibrationVar)
    calibrationEntry.place(x=15, y=65)

    def continue_callback():
        app.destroy()

    continueButton = tk.Button(app, text="Continue", font=20, fg="blue", command=continue_callback)
    continueButton.place(x=460, y=428)

    def option_callback(*args):
        sourceEntry.configure(state="normal")
        selectedOption = optionVar.get()

        if selectedOption == "Laptop webcam":
            sourceLabel.configure(text="Don't worry about this")
            sourceVar.set("0")
            sourceEntry.configure(state="disabled")
            calibrationVar.set("webcam_calibration_matrix.yml")

        elif selectedOption == "IP webcam":
            sourceLabel.configure(text="Insert the webcam IP")
            sourceVar.set("https://192.168.1.37:8080/video")
            calibrationVar.set("phone_calibration_matrix.yml")

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
