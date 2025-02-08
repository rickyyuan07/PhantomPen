import tkinter as tk


class SimpleGUI:
    def __init__(self, window, is_authenticated):
        """Initialize a basic Tkinter window."""
        self.window = window
        self.window.title("Authentication Status")
        self.window.geometry("400x200")  # Set window size

        # Determine authentication message based on the argument
        if is_authenticated:
            message = "✅ Authenticated! Welcome"
            color = "green"
        else:
            message = "❌ Authentication failed!"
            color = "red"

        # Label
        self.label = tk.Label(window, text=message, font=("Arial", 14), fg=color)
        self.label.pack(pady=20)

        # Button to close the window
        self.quit_button = tk.Button(window, text="Close", command=window.quit)
        self.quit_button.pack(pady=10)

    