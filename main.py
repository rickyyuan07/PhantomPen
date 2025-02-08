import os
from simple_draw import PhantomPen
from authenticator import SignatureAuth

if __name__ == "__main__":
    # app = PhantomPen()
    # app.run()
    auth = SignatureAuth()
    for i in range(10):
        npy1_path = os.path.join("signatures", "daniel", f"{i}.npy")
        npy2_path = os.path.join("signatures", "ricky", f"{i}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Daniel-Ricky Cosine Similarity: {similarity_score:.4f}")
    for i in range(9):
        npy1_path = os.path.join("signatures", "daniel", f"{i}.npy")
        npy2_path = os.path.join("signatures", "daniel", f"{i+1}.npy")

        similarity_score = auth.compare_npy(npy1_path, npy2_path)
        print(f"Daniel-Daniel+1 Cosine Similarity: {similarity_score:.4f}")
    