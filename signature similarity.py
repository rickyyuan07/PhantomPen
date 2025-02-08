import cv2
from skimage.metrics import structural_similarity as ssim

def compare_signatures(signature_path1, signature_path2):
    """
    Compares two signature images using structural similarity index (SSIM).

    Args:
        signature_path1 (str): Path to the first signature image.
        signature_path2 (str): Path to the second signature image.

    Returns:
        float: Similarity score between the two images (1.0 means identical, lower values indicate differences).
               Returns None if an error occurs during image loading or processing.
    """
    try:
        img1 = cv2.imread(signature_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(signature_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            print("Error: Could not open or find the images!")
            return None
        
        # Ensure both images have the same dimensions
        if img1.shape != img2.shape:
            print("Resizing images to the same dimensions...")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        similarity_score = ssim(img1, img2)
        return similarity_score

    except Exception as e:
         print(f"An error occurred: {e}")
         return None

# Example usage:
signature_file1 = "signature1.png"
signature_file2 = "signature2.png"

similarity = compare_signatures(signature_file1, signature_file2)

if similarity is not None:
    print(f"Similarity between signatures: {similarity:.4f}")

    if similarity >= 0.8:
        print("The signatures are likely a match.")
    else:
        print("The signatures are likely different.")