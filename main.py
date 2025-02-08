import os
from simple_draw import PhantomPen
import argparse
import tkinter as tk
from simple_gui import SimpleGUI


def start_gui(is_authenticated):
    """Start the GUI application."""
    window = tk.Tk()
    app = SimpleGUI(window, is_authenticated)
    window.mainloop()


if __name__ == "__main__":
   
    # from simple draw
    parser = argparse.ArgumentParser(description="Simple draw & signature collection app")
    parser.add_argument("-n", "--name", type=str, default="user", help="the name of the user")
    parser.add_argument("-s", "--signature_dir", type=str, default="test", help="Directory to store signatures")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="glow", help="Drawing style")
    parser.add_argument("-p", "--phantom", action="store_true", help="Enable phantom effect")
    # Add necessary arguments
    # TODO
    # arguments
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the trained model checkpoint (e.g., siamese_signature_model.pth)")
    parser.add_argument("--real_dir", type=str, default="signatures/rickyy", help="Path to the real signature directory")
    parser.add_argument("--fake_dir", type=str, default="signatures/ricky", help="Path to the fake signature directory")

    args = parser.parse_args()
    
    user_dir = os.path.join(args.signature_dir, args.name)
    os.makedirs(user_dir, exist_ok=True)  # Create directory if it doesn't exist

    # files = [f for f in os.listdir(user_dir) if f.endswith(".npy")]
    # args.signature_idx = 0
    # if files:
    #     args.signature_idx = max([int(os.path.splitext(f)[0]) for f in files]) + 1
    # print(args.signature_idx)


    # Get user name from user
    # Some GUI code to get the user name
    # TODO

    app = PhantomPen(args)
    app.run()

    # Run the Authenticator with the user's signature
    # TODO

    # Pop-up login success or failure message (GUI)
    # TODO
    authenticated = True 
    start_gui(authenticated)
    