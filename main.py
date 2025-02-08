import os
from simple_draw import PhantomPen
import argparse
import tkinter as tk
from simple_gui import SimpleGUI
from authenticator import SignatureAuth


def start_gui(is_authenticated):
    """Start the GUI application."""
    window = tk.Tk()
    app = SimpleGUI(window, is_authenticated)
    window.mainloop()


if __name__ == "__main__":
   
    # from simple draw
    parser = argparse.ArgumentParser(description="Simple draw & signature collection app")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="glow", help="Drawing style")
    parser.add_argument("-p", "--phantom", action="store_true", help="Enable phantom effect")
    # Add necessary arguments
    # TODO
    # arguments
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the trained model checkpoint (e.g., siamese_signature_model.pth)")
    parser.add_argument("-n", "--name", type=str, default="", help="name to attempt auth")
    parser.add_argument("--base_dir", type=str, default="signatures", help="Path to the signature prototypes")
    parser.add_argument("--proto_dir", type=str, default="signatures/prototypes", help="Path to the signature prototypes")
    args = parser.parse_args()
    args.signature_dir = "./test/"
    args.signature_idx = 0
    args.pipeline = True

    # Get user name from user
    # Some GUI code to get the user name
    # TODO
    args.name = input("Enter your name: ")
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

    # Pop-up login success or failure message (GUI)
    # TODO
    start_gui(authenticated)
    
