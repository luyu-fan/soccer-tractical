from ttkthemes import ThemedTk

from lib.interface.frames.app import App

if __name__ == "__main__":

    # main_window = tkinter.Tk()
    main_window = ThemedTk(theme="equilux")

    app = App(main_window, 1280, 720)
    app.init(finished_file="./datasets/record/finished.txt")   # TODO replace by DB

    main_window.mainloop()

    # spliting_videos("../")

