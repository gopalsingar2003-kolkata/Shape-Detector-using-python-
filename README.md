# Real-Time Shape & Finger Detector

A Python application that uses your computer's webcam to detect and classify geometric shapes and count fingers in real-time using OpenCV.

## 🚀 Features

### Shape Detection
- **Real-time Detection:** Processes live video feed from your webcam.
- **Shape Classification:** Identifies Triangles, Squares, Rectangles, Pentagons, Hexagons, and Circles.
- **Visual Feedback:** Draws contours around detected shapes and labels them with their name.
- **Noise Reduction:** Filters out small contours to prevent flickering and false detections.
- **Traditional CV Approach:** Uses contour approximation and geometric properties instead of heavy machine learning models.

### Finger Detection
- **Hand Detection:** Uses HSV color space skin detection to identify hand regions.
- **Finger Counting:** Estimates finger count using convex hull and convexity defects analysis.
- **Visual Feedback:** Draws hand contour, convex hull, and finger gap points.
- **Real-time Overlay:** Displays finger count on the video feed.

### General Features
- **FPS Counter:** Real-time frame rate display.
- **Toggle Controls:** Enable/disable shape detection (S) and finger detection (F) independently.
- **Information Panel:** On-screen display showing FPS, finger count, and detection modes.

## 🛠️ Technologies Used

- **Python 3.x** - Programming language
- **OpenCV (`opencv-python`)** - Computer vision library for image processing
- **NumPy** - Numerical computing for array operations
- **Math & Time** - Standard libraries for calculations and timing

## 📦 Installation

1. **Clone or create the project folder:**
   ```bash
   mkdir shape_detector
   cd shape_detector
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 How to Run in VS Code

1. Open the `shape_detector` folder in VS Code.
2. Open a new terminal (`Ctrl + Shift + `` ` ``).
3. Ensure your virtual environment is activated (see installation steps).
4. Run the program:
   ```bash
   python main.py
   ```

## 📷 Webcam Permissions

- **Windows:** Ensure that "Camera access" is turned on in **Settings > Privacy & security > Camera**.
- **macOS:** When you run the script, macOS may prompt you to allow VS Code or Terminal to access the camera. Click **OK**.
- **Linux:** Ensure your user is part of the `video` group: `sudo usermod -aG video $USER`.

## 🔍 How Shape Detection Works

The program follows these image processing steps:

1. **Grayscale Conversion:** Converts the colorful camera feed to grayscale to simplify processing.
2. **Gaussian Blur:** Smooths the image to reduce high-frequency noise.
3. **Canny Edge Detection:** Identifies the edges of objects in the frame.
4. **Contour Finding:** Connects the edges to find closed shapes.
5. **Contour Approximation (`cv2.approxPolyDP`):** Simplifies a complex contour into a polygon with a smaller number of vertices. This is the key to identifying shapes:
   - 3 vertices → Triangle
   - 4 vertices → Square or Rectangle
   - 5 vertices → Pentagon
   - 6 vertices → Hexagon
   - Many vertices → Circle
6. **Geometric Analysis:** For 4-sided shapes, the program calculates the aspect ratio (width/height) to distinguish between a square (ratio ≈ 1) and a rectangle.

## 🖐️ How Finger Detection Works

1. **Color Space Conversion:** Converts BGR frame to HSV for better skin color segmentation.
2. **Skin Mask Creation:** Applies HSV threshold range to isolate skin-colored pixels.
3. **Morphological Operations:** Opening and closing to remove noise and fill gaps in the mask.
4. **Hand Contour Detection:** Finds the largest contour in the mask (assumed to be the hand).
5. **Convex Hull:** Computes the convex hull of the hand contour.
6. **Convexity Defects:** Finds indentations in the hull (gaps between fingers).
7. **Angle & Depth Analysis:** Filters defects by angle (< 90°) and depth (> 10px) to count finger gaps.
8. **Finger Count:** Finger count = finger gaps + 1 (clamped to 0-5).

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
| **S** | Toggle Shape Detection (ON/OFF) |
| **F** | Toggle Finger Detection (ON/OFF) |
| **R** | Reset FPS timer |

## 🛠️ Troubleshooting

- **Webcam not opening:**
  - Ensure no other app (Zoom, Teams, etc.) is using the camera.
  - Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `main.py` if you have multiple cameras.

- **Shapes not detected:**
  - Use a simple, contrasting background (e.g., a white piece of paper on a dark table).
  - Ensure there is good lighting.

- **Too much noise:**
  - Adjust the `MIN_SHAPE_AREA` constant in `main.py` to ignore smaller objects.

- **Finger detection not working:**
  - Ensure your hand is clearly visible with good lighting.
  - Skin detection works best with standard lighting conditions.
  - Adjust `SKIN_LOWER` and `SKIN_UPPER` HSV ranges in `main.py` for different skin tones.

## 📁 Project Structure

```
shape_detector/
├── main.py          # Main application with shape & finger detection
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.