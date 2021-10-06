import os, sys
import traceback

__all__ = ['TeeLog']

# Context manager that copies stdout and any exceptions to a log file
class TeeLog(object):
    """
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, filename = None):
        self.files = []
        if filename is not None: self.append(filename)
        self.stdout = sys.stdout
    # __init__

    def append(self, filename):
        try:
            file = open(filename, 'w') if filename is not None and filename != "" else None
        except:
            file = None
        # try
        if file is not None: self.files.append(file)
    # init

    def __enter__(self):
        sys.stdout = self
        return self
    # __enter__

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            for file in self.files: file.write(traceback.format_exc())
        for file in self.files: file.close()
    # __exit__

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()
        self.stdout.write(data)
    # write

    def flush(self):
        for file in self.files: file.flush()
        self.stdout.flush()
    # flush
# TeeLog