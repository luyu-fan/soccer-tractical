import tkinter
from ttkthemes import ThemedTk

from lib.interface.frames.app import App

if __name__ == "__main__":

    # main_window = tkinter.Tk()
    main_window = ThemedTk(theme="equilux")
    main_window.title("SoccerDetector")
    main_window.iconbitmap("./assets/soccer_roteiro.ico")

    app = App(main_window, 1280, 720)
    app.init()

    main_window.mainloop()

    # spliting_videos("../")

