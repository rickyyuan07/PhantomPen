import os
from simple_draw import PhantomPen
import argparse

if __name__ == "__main__":
    # app = PhantomPen()
    # app.run()
    # arguments
    parser = argparse.ArgumentParser(description="Compare signature similarities using a trained model.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the trained model checkpoint (e.g., siamese_signature_model.pth)")
    parser.add_argument("--real_dir", type=str, default="signatures/rickyy", help="Path to the real signature directory")
    parser.add_argument("--fake_dir", type=str, default="signatures/ricky", help="Path to the fake signature directory")
   
    # from simple draw
    parser.add_argument("-n", "--name", type=str, default="user", help="the name of the user")
    parser.add_argument("-s", "--signature_dir", type=str, default="signatures", help="Directory to store signatures")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="glow", help="Drawing style")
    parser.add_argument("-p", "--phantom", action="store_true", help="Enable phantom effect")
    parser = argparse.ArgumentParser(description="Simple draw & signature collection app")
    parser.add_argument("-n", "--name", type=str, default="user", help="the name of the user")
    parser.add_argument("-s", "--signature_dir", type=str, default="test", help="Directory to store signatures")
    parser.add_argument("-st", "--style", type=str, choices=["glow", "neon_blue", "fire"], default="glow", help="Drawing style")
    parser.add_argument("-p", "--phantom", action="store_true", help="Enable phantom effect")
    # Add necessary arguments
    # TODO

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
    
    user_dir = os.path.join(args.signature_dir, args.name)
    os.makedirs(user_dir, exist_ok=True)  # Create directory if it doesn't exist

    files = [f for f in os.listdir(user_dir) if f.endswith(".npy")]
    args.signature_idx = 0
    if files:
        args.signature_idx = max([int(os.path.splitext(f)[0]) for f in files]) + 1
    print(args.signature_idx)

    # display GUI 

    




    app = PhantomPen(args)
    app.run()
    # done with simple draw





    # to decide the similarity 
    # auth = SignatureAuth(args.ckpt_path)
    # for i in range(17):
    #     npy1_path = os.path.join("signatures", "rickyy", f"{i}.npy")
    #     npy2_path = os.path.join("signatures", "rickyy", f"{i+1}.npy")

    #     similarity_score = auth.compare_npy(npy1_path, npy2_path)
    #     print(f"Rickyy-Rickyy Cosine Similarity: {similarity_score:.4f}")

    # for i in range(18):
    #     npy1_path = os.path.join("signatures", "ricky", f"{i}.npy")
    #     npy2_path = os.path.join("signatures", "rickyy", f"{i}.npy")

    #     similarity_score = auth.compare_npy(npy1_path, npy2_path)
    #     print(f"Ricky-Rickyy Cosine Similarity: {similarity_score:.4f}")
    
    # for i in range(17):
    #     npy1_path = os.path.join("signatures", "ricky", f"{i}.npy")
    #     npy2_path = os.path.join("signatures", "ricky", f"{i+1}.npy")

    #     similarity_score = auth.compare_npy(npy1_path, npy2_path)
    #     print(f"Ricky-Ricky Cosine Similarity: {similarity_score:.4f}")

    # for i in range(17):
    #     npy1_path = os.path.join("signatures", "ricky", f"{30}.npy")
    #     npy2_path = os.path.join("signatures", "rickyy", f"{i}.npy")

    #     similarity_score = auth.compare_npy(npy1_path, npy2_path)
    #     print(f"Ricky_test-Rickyy Cosine Similarity: {similarity_score:.4f}")