import os
from simple_draw import PhantomPen
import argparse
import tkinter as tk
from tkinter import messagebox, simpledialog
# from simple_gui import SimpleGUI
from authenticator import SignatureAuth


def start_gui(is_authenticated):
    """Start the GUI application and show authentication result."""
    window = tk.Tk()
    window.withdraw()  # Hide the root window

    if is_authenticated:
        messagebox.showinfo("Authentication", "Login Successful! 🎉")
    else:
        messagebox.showerror("Authentication", "Login Failed! ❌")

    window.destroy()


def get_username():
    """Prompt the user for their name using a GUI input box."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    name = simpledialog.askstring("User Authentication", "Enter your name:")
    root.destroy()
    return name


if __name__ == "__main__":
   
    # from simple draw
    parser = argparse.ArgumentParser(description="Simple draw & signature collection app")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="fire", help="Drawing style")
    parser.add_argument("-p", "--phantom", action="store_true", help="Enable phantom effect")
    # Add necessary arguments
    # TODO
    # arguments
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the trained model checkpoint (e.g., siamese_signature_model.pth)")
    parser.add_argument("-n", "--name", type=str, default="", help="name to attempt auth")
    parser.add_argument("--base_dir", type=str, default="signatures", help="Path to the signature prototypes")
    parser.add_argument("--proto_dir", type=str, default="signatures/prototypes", help="Path to the signature prototypes")
    args = parser.parse_args()
    args.signature_dir = "./test/"  # Temporary directory for signature collection
    args.signature_idx = 0
    args.pipeline = True  # Ensure pipeline mode is enabled

    # Get user name via GUI input
    args.name = get_username()
    if not args.name:  # If the user cancels input, exit
        print("No name entered. Exiting...")
        exit(1)

    user_dir = os.path.join(args.signature_dir, args.name)
    os.makedirs(user_dir, exist_ok=True)  # Create directory if it doesn't exist

    app = PhantomPen(args)
    app.run()

    # Run the Authenticator with the user's signature
    
    auth = SignatureAuth(args.ckpt_path)
    # proto_path = os.path.join(args.proto_dir, f"{args.name}.npy")
    # auth.challenge_proto(proto_path, args.new_img_path)
    representative_npy_path = os.path.join(args.base_dir, 'train', 'real', args.name, "0.npy")
    new_img_path = os.path.join(user_dir, "0.npy")
    authenticated = auth.challenge_npy(representative_npy_path, new_img_path)

    # Show authentication result in a pop-up window
    start_gui(authenticated)
    
