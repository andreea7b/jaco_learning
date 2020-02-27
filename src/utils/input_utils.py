import Tkinter as tk
from threading import Thread

class KeyboardListener(Thread):
    def __init__(self, bindings):
        """
        Handles key presses in its own thread. Must be used by calling .start()
        on the object, which will then call run() internally

        Params:
            bindings: list of (key, fn) pairs
        """
        self.bindings = bindings

    def run(self):
        self.main = tk.Tk()
        for key, fn in self.bindings:
            self.main.bind(key, fn)
        self.main.mainloop()
