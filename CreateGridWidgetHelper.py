import tkinter as tk
from tkinter import ttk

class CreateGridWidgetHelper:
    
    @staticmethod
    def tk_frame(root, position=(0,0), rowspan=1, colunmspan=1, sticky="news", rowconfigure=None, columnconfigure=None, relief=None):
        colunm, row = position
        tk_frame = tk.Frame(root, relief=relief)
        tk_frame.grid(row=row, column=colunm, rowspan=rowspan, columnspan=colunmspan, sticky="news")
        if rowconfigure is not None:
            tk_frame.rowconfigure(rowconfigure, weight=1)
        if columnconfigure is not None:
            tk_frame.columnconfigure(columnconfigure, weight=1)
        return tk_frame
    
    @staticmethod
    def canvas(root, background="#222222", position=(0,0), columnspan=1):
        colunm, row = position
        canvas = tk.Canvas(root, background=background)
        canvas.grid(row=row, column=colunm, columnspan=columnspan, sticky="news")
        return canvas
    
    @staticmethod
    def ttk_button(root, text, command=None, position=(0,0)):
        colunm, row = position
        button = ttk.Button(root, text=text, command=command)
        button.grid(row=row, column=colunm)
        return button
    
    @staticmethod
    def ttk_label_and_entry(root, label_text, position=(0,0), sticky="ew"):
        column, row = position
        label = ttk.Label(root, text=label_text)
        label.grid(row=row, column=column)
        entry = ttk.Entry(root)
        entry.grid(row=row, column=column+1, sticky=sticky)
        entry.insert(0, "")
        return entry
    
    def tk_label(root, text="", bg="#222222", fg="black", font=("Meiryo", 12, "bold"), width=10, height=3, relief="solid", bd=2, position=(0,0), columnspan=1, sticky=None):
        column, row = position
        label = tk.Label(
            root,
            text=text,
            bg=bg,
            fg=fg,
            font=font,
            width=width,
            height=height,
            relief=relief,
            bd=bd
        )
        label.grid(row=row, column=column, columnspan=columnspan, sticky=sticky)

        return label
