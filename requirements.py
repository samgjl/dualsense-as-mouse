import os

def main():
    print("Creating virtual environment...\n")
    os.system("python -m venv venv")

    print("Installing requirements...\n")
    requirements = ["pydualsense", "pyautogui", "numpy", "matplotlib", "scipy"] # DO I NEED ALL OF THESE???
    os.system("pip install " + " ".join(requirements) + " --target=.\\venv\\Lib\\site-packages")
    os.system("copy hidapi.dll .\\venv\\Lib\\site-packages")

    print("\nDone!")

if __name__ == "__main__":
    main()