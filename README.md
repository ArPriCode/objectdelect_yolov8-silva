# YOLOv8-Silva Object Detection Project

A comprehensive YOLOv8 implementation for object detection, image classification, and real-time video processing using Python and OpenCV.

## 🚀 Features

- **Object Detection**: Detect 80+ COCO classes with high accuracy
- **Real-time Video Processing**: Live detection on video streams and webcam
- **Image Classification**: Single image processing with bounding boxes
- **Customizable Confidence**: Adjustable detection thresholds
- **Multiple Input Sources**: Images, videos, and webcam support
- **Visual Output**: Bounding boxes with class labels and confidence scores

## 📋 Tech Stack

- **Python 3.7+**: Core programming language
- **Ultralytics YOLOv8**: State-of-the-art object detection framework
- **PyTorch**: Deep learning backend
- **OpenCV**: Computer vision and video processing
- **NumPy**: Numerical computations and array operations
- **SciPy**: Scientific computing utilities
- **Pillow**: Image processing and I/O
- **Matplotlib**: Visualization and plotting

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd yolov8-silva-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 model (automatically downloaded on first run):
```bash
# Model will be saved to weights/yolov8n.pt
```

## 📁 Project Structure

```
yolov8-silva-main/
├── datasets/
│   └── animals/           # Custom animal classification dataset
│       ├── train/         # Training images
│       └── test/          # Test images
├── inference/
│   ├── images/           # Sample images for testing
│   ├── videos/           # Sample videos for testing
│   └── banner.png        # Project banner
├── utils/
│   └── coco.txt          # COCO class names (80 classes)
├── weights/
│   ├── yolov8n.pt        # Pretrained YOLOv8 nano model
│   └── read.txt          # Model information
├── yolov8_basics.py      # Basic image detection script
├── yolov8_n_opencv.py    # Real-time video detection script
├── train_classification.py # Custom model training script
├── yolo_check_system.py  # System compatibility check
└── requirements.txt       # Python dependencies
```

## 🎯 Usage

### 1. Basic Image Detection

Run object detection on a single image:

```bash
python yolov8_basics.py
```

**What it does:**
- Loads YOLOv8n model
- Processes `inference/images/img0.JPG`
- Detects objects with 25% confidence threshold
- Saves results to `runs/detect/predict/`
- Prints detection results

### 2. Real-time Video Detection

Run real-time detection on video or webcam:

```bash
python yolov8_n_opencv.py
```

**Features:**
- Processes video from `inference/videos/afriq0.MP4`
- Real-time bounding box drawing
- Class labels with confidence scores
- Press 'q' to quit
- Random colors for each class

**To use webcam instead:**
Edit line 87 in `yolov8_n_opencv.py`:
```python
cap = cv2.VideoCapture(0)  # 0 for default webcam
```

### 3. Custom Model Training

Train on custom dataset:

```bash
python train_classification.py
```

## 🔧 Configuration

### Detection Parameters

**Confidence Threshold:**
```python
# In yolov8_basics.py
detection_output = model.predict(source="image.jpg", conf=0.25)

# In yolov8_n_opencv.py  
detect_params = model.predict(source=[frame], conf=0.45)
```

**Model Selection:**
```python
# Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO("weights/yolov8n.pt", "v8")
```

### Video Settings

**Frame Resolution:**
```python
# In yolov8_n_opencv.py
frame_wid = 640
frame_hyt = 480
```

## 📊 Supported Classes (COCO Dataset)

The model can detect 80 object classes including:
- **People**: person
- **Vehicles**: car, motorcycle, bus, truck, bicycle, airplane, train, boat
- **Animals**: cat, dog, horse, cow, elephant, bear, zebra, giraffe, sheep, bird
- **Objects**: laptop, keyboard, mouse, cell phone, book, clock, vase, scissors
- **Furniture**: chair, couch, bed, dining table, toilet
- **Food**: banana, apple, sandwich, pizza, cake, donut
- And many more...

## 🎨 Customization

### Adding New Classes

1. Update `utils/coco.txt` with your class names
2. Retrain the model on your custom dataset
3. Update the class list in your scripts

### Changing Colors

```python
# In yolov8_n_opencv.py
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255) 
    b = random.randint(0,255)
    detection_colors.append((b,g,r))
```

### Custom Input Sources

**Image:**
```python
detection_output = model.predict(source="path/to/your/image.jpg")
```

**Video:**
```python
cap = cv2.VideoCapture("path/to/your/video.mp4")
```

**Webcam:**
```python
cap = cv2.VideoCapture(0)  # 0, 1, 2 for different cameras
```

## 📈 Performance

- **Speed**: ~30-70ms per frame (depending on hardware)
- **Accuracy**: High precision on COCO dataset
- **Model Size**: YOLOv8n (~6MB) - lightweight and fast
- **Memory**: Low GPU/CPU usage

## 🐛 Troubleshooting

### Common Issues

**1. Import Error:**
```bash
pip install ultralytics
```

**2. CUDA/GPU Issues:**
```python
# Check if CUDA is available
import torch
print(torch.cuda.is_available())
```

**3. Video Not Opening:**
- Check video file path
- Ensure video format is supported (MP4, AVI, MOV)
- Try different camera index (0, 1, 2)

**4. Low FPS:**
- Reduce frame resolution
- Use smaller model (yolov8n.pt)
- Lower confidence threshold

## 📝 Examples

### Detection Results
```
image 1/1: 384x640 1 person, 1 bicycle, 2 cups, 2 potted plants, 1 dining table, 1 laptop, 1 keyboard, 50.2ms
Speed: 1.6ms preprocess, 50.2ms inference, 1.5ms postprocess per image at shape (1, 3, 384, 640)
```

### Output Files
- Detection results saved to `runs/detect/predict/`
- Annotated images with bounding boxes
- Confidence scores and class labels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [COCO Dataset](https://cocodataset.org/) for training data
- OpenCV community for computer vision tools

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the [Ultralytics documentation](https://docs.ultralytics.com/)
- Review OpenCV documentation for video processing

---

**Happy Detecting! 🎯**
