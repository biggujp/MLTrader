from pywinauto import Desktop

for w in Desktop(backend="uia").windows():
    print(w.window_text())