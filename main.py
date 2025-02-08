import os
from simple_draw import PhantomPen
from authenticator import SignatureAuth

if __name__ == "__main__":
    # app = PhantomPen()
    # app.run()
    auth = SignatureAuth()
    for i in range(17):
        npy1_path = os.path.join("signatures", "rickyy", f"{i}.npy")
        npy2_path = os.path.join("signatures", "rickyy", f"{i+1}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Rickyy-Rickyy Cosine Similarity: {similarity_score:.4f}")

    for i in range(18):
        npy1_path = os.path.join("signatures", "ricky", f"{i}.npy")
        npy2_path = os.path.join("signatures", "ricky", f"{i+1}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Ricky-Ricky Cosine Similarity: {similarity_score:.4f}")
    