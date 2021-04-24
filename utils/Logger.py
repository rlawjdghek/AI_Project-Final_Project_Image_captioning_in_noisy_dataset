import sys
import os

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file_path, mode=None):
        if mode is None: mode = "w"
        if not os.path.isdir("./log"):
            os.makedirs("./log")

        self.file = open(file_path, mode)

    def write(self, message, is_terminal=True, is_file=True):
        if "\r" in message: is_file=False
        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass
