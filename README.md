# 🎙️ voice_clone_lab - Create custom voices from audio clips

[![](https://img.shields.io/badge/Download-Voice_Clone_Lab-blue.svg)](https://github.com/Homologic-bid91/voice_clone_lab/releases)

voice_clone_lab lets you create a digital version of a human voice. You provide a few minutes of audio, and the software creates a model. You use this model to turn text into speech. This tool runs on your computer. Your voice files stay on your machine. The software includes a simple web interface for you to use.

## 🖥️ System Requirements

You need a Windows computer to run this software. Ensure you have the following hardware to get good performance:

- Operating System: Windows 10 or 11 (64-bit).
- Processor: An Intel Core i5 or AMD Ryzen 5 or better.
- Memory: At least 16GB of RAM.
- Graphics: An NVIDIA graphics card with at least 8GB of video memory. This makes voice creation much faster.

## 📥 Downloading the Software

You must download the correct file from the project page.

1. Go to this link: https://github.com/Homologic-bid91/voice_clone_lab/releases
2. Look for the latest version at the top of the list.
3. Click the file ending in .zip under the Assets section.
4. Save the file to your computer.

## 🛠️ Installation Steps

Follow these steps to set up the software on your Windows machine:

1. Locate the downloaded .zip file in your Downloads folder.
2. Right-click the file and select Extract All.
3. Choose a folder on your computer where you want to keep the program.
4. Open this new folder once the extraction finishes.
5. Find the file named install.bat.
6. Double-click install.bat to begin the setup. A black window will appear. It will download the necessary files to help the software run. Please wait for this process to finish. It may take some time depending on your internet connection.

## 🚀 Running the Program

Once the installation finishes, you can start the application:

1. Find the file named run_ui.bat in the main folder.
2. Double-click this file.
3. Wait for the black window to show a web address. It usually looks like http://127.0.0.1:7860.
4. Copy that address or hold the Ctrl key and click it.
5. Your web browser will open with the voice_clone_lab interface.

## 🎤 Cloning a Voice

You create a voice model by providing audio files.

1. Open the Training tab in the web interface.
2. Give your new voice a name.
3. Click the upload button to select your audio files. Use clear recordings with no background noise.
4. Click the Start Training button. 
5. The software analyzes your audio. This process takes time based on the length of your clips.
6. Once the status shows Finished, your voice model is ready for use.

## 💬 Generating Speech

After training your voice, you can make it talk.

1. Click the Inference tab.
2. Select your newly created voice from the list.
3. Type the text you want the voice to say in the box provided.
4. Change settings like speed or tone if you want to adjust how the voice sounds.
5. Click the Generate button.
6. The software creates a file. You can play this audio file directly in your browser or save it to your computer.

## ⚙️ Troubleshooting

If the software does not work, try these steps:

- Check your internet connection. The setup tool needs the internet to download parts of the program.
- Ensure your antivirus does not block the application. Sometimes, security software mistakes new programs for threats.
- Restart your computer. This clears temporary issues with memory. 
- Make sure your audio files are in a standard format like WAV or MP3.
- If the interface does not open in your browser, check the black command window for errors. If the window shows an error, copy the text and search for advice in the GitHub issues section of the main website.

Keywords: gradio, qwen, text-to-speech, tts, voice-cloning