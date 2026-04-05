import tkinter as tk

def click():
    global cookies
    cookies += 1
    label.config(text=f"Cookies: {cookies}")

cookies = 0
root = tk.Tk()
label = tk.Label(root, text="Cookies: 0", font=("Arial", 20))
label.pack()
btn = tk.Button(root, text="Click Me!", command=click)
btn.pack()
root.mainloop()