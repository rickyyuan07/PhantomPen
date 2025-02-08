import os
from simple_draw import PhantomPen
from authenticator import SignatureAuth
import argparse

if __name__ == "__main__":
    # app = PhantomPen()
    # app.run()
    parser = argparse.ArgumentParser(description="Compare signature similarities using a trained model.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the trained model checkpoint (e.g., siamese_signature_model.pth)")
    parser.add_argument("--real_dir", type=str, default="signatures/rickyy", help="Path to the real signature directory")
    parser.add_argument("--fake_dir", type=str, default="signatures/ricky", help="Path to the fake signature directory")
    args = parser.parse_args()

    auth = SignatureAuth(args.ckpt_path)
    for i in range(17):
        npy1_path = os.path.join("signatures", "rickyy", f"{i}.npy")
        npy2_path = os.path.join("signatures", "rickyy", f"{i+1}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Rickyy-Rickyy Cosine Similarity: {similarity_score:.4f}")

    for i in range(18):
        npy1_path = os.path.join("signatures", "ricky", f"{i}.npy")
        npy2_path = os.path.join("signatures", "rickyy", f"{i}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Ricky-Rickyy Cosine Similarity: {similarity_score:.4f}")
    
    for i in range(17):
        npy1_path = os.path.join("signatures", "ricky", f"{i}.npy")
        npy2_path = os.path.join("signatures", "ricky", f"{i+1}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Ricky-Ricky Cosine Similarity: {similarity_score:.4f}")

    for i in range(17):
        npy1_path = os.path.join("signatures", "ricky", f"{30}.npy")
        npy2_path = os.path.join("signatures", "rickyy", f"{i}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Ricky_test-Rickyy Cosine Similarity: {similarity_score:.4f}")