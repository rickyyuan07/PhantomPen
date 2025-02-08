# PhantomPen: Free-Hand Signature Authentication
*Tartan Hackathon 2025 at Carnegie Mellon University.*
PhantomPen is an innovative application that enables **signature authentication using hand gestures**. Users can draw their signature on-screen using their fingers, which serves as their biometric authentication method.

## ✨ How It Works
### 1.	Signature Collection:
- Users first draw multiple samples of their signature.
- This serves as the enrollment phase for authentication.
### 2.	Authentication (Login):
- The user enters their username.
- They draw their signature on the screen.
- A machine learning model verifies the signature.
### 3.	Result:
- ✅ Match found: Access granted.
- ❌ Signature does not match: Authentication fails.

## 🛠 Pre-requisites
- Python 3.6 or higher

## 🚀 Installation
Ensure you have conda installed. If not, install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
```
conda create -n PhantomPen python=3.10
conda activate PhantomPen
pip install -r requirements.txt
```
### Note
This project is tested on macOS. If you’re using Windows, you may experience GUI issues, but the core functionality will work fine.

## ✍️ Signature Collection
To collect signature samples, run:
```
python3 simple_draw.py --name <your_name> --style <style> -p
```
- your_name: Your username
- style: Choose from "glow", "neon_blue", or "fire"
- -p: Enables the Phantom effect 🤩

### How to Draw
- Touch your thumb and index fingertips together: Start drawing ✏️
- Separate your fingertips: Stop drawing
- Press s: Save the signature
- Press c: Clear the canvas
- Press q: Quit

## 🔑 Signature Login
To run the main script
```
python3 main.py --style <style> -p
```

Then follow the GUI prompts to authenticate your signature. 🖋️

## 👨‍💻 Developers & Contributors

Project created for Tartan Hackathon 2025 🏆 at Carnegie Mellon University.

For questions or contributions, feel free to open an issue or submit a pull request! 🚀