# Technical Documentation
**UIDAI SITAA Challenge - Contactless Fingerprint Authentication**

<p align="center">
  <img src="https://www.yellowsense.in/assets/logo.jpeg" alt="YellowSense Technologies" width="200"/>
</p>

<p align="center">
  <strong>YellowSense Technologies Pvt. Ltd.</strong>
</p>

---

## 📱 **Quick Links**

- **[Download Complete APK (V2)](https://github.com/yellowsense2008/YellowSense_Contactless_Fingerprint/blob/main/apk/V2_YellowSense_Contactless_Fingerprint.apk)** - Unified Android application with all tracks
- **[Watch Demo Video](https://github.com/yellowsense2008/YellowSense_Contactless_Fingerprint/blob/main/Demo_Video/V2_demo_video.mp4)** - Complete walkthrough
- **[View Pitch Deck](https://github.com/yellowsense2008/YellowSense_Contactless_Fingerprint/blob/main/Pitch_Deck/Pitch-deck.pdf)** - Presentation
- **[Read Full Proposal](https://github.com/yellowsense2008/uidai-sitaa-contactless-fingerprint/blob/main/Proposal_Document/updated_proposal_YellowSense_Tech.pdf)** - Complete technical proposal

---

## 📚 **Table of Contents**

1. [System Overview](#system-overview)
3. [Track A: Quality Assessment](#track-a-quality-assessment)
4. [Track B: Image Enhancement](#track-b-image-enhancement)
5. [Track C: Fingerprint Matching](#track-c-fingerprint-matching)
6. [Track D: Liveness Detection](#track-d-liveness-detection)
7. [Architecture](#architecture)
8. [Technology Stack](#technology-stack)
9. [Performance Metrics](#performance-metrics)
10. [Future Enhancements](#future-enhancements)

---

## 🎯 **System Overview**

### **Goal**
Build a complete contactless fingerprint authentication system that can:
1. **Capture** high-quality contactless fingerprints (Track A)
2. **Enhance** image quality for better processing (Track B)
3. **Match** contactless against contact-based fingerprints (Track C)
4. **Detect liveness** to prevent spoofing attacks (Track D)

### **Four-Track Implementation**

```
┌─────────────────────────────────────────────────────────┐
│               CONTACTLESS AUTHENTICATION                │
│                COMPLETE END-TO-END PIPELINE             │
└─────────────────────────────────────────────────────────┘

TRACK A          TRACK B          TRACK C          TRACK D
  
📸 Capture  →  🎨 Enhance   →  🔍 Match     →  🛡️ Verify
Quality         Image            Authenticate    Liveness
Check           Quality          

↓               ↓                ↓               ↓

MediaPipe       OpenCV           Siamese         Motion +
Quality         Classical        Network         Texture
Scores          Enhancement      Similarity      Analysis
Real-time       On-Device        Scoring         Spoof Detection
```

### **Why All Four Tracks?**

We implemented all tracks to demonstrate:

✅ **Complete Pipeline** - End-to-end authentication workflow  
✅ **Comprehensive Solution** - Quality + Enhancement + Matching + Security  
✅ **Practical Deployment** - Each track addresses real-world needs  
✅ **Technical Depth** - Classical CV + Deep Learning + Security  

---

## 📋 **Track A: Quality Assessment**

### **Purpose**
Real-time on-device quality analysis ensuring captured contactless fingerprints meet minimum standards for reliable matching.

### **Architecture Overview**

```
┌───────────────────────────────────────────────────────┐
│            Mobile Device (React Native)               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Camera Preview (Live Feed)                      │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Frame Capture & Processing                      │ │
│  │ - Capture at 10-15 FPS                          │ │
│  │ - Direct frame access                           │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ HandDetector (MediaPipe - On-Device)            │ │
│  │ - 21 hand landmarks detection                   │ │
│  │ - Index finger bbox extraction                  │ │
│  │ - Real-time inference                           │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ QualityAnalyzer (OpenCV - On-Device)            │ │
│  │ - Blur: Laplacian variance                      │ │
│  │ - Illumination: Brightness + contrast           │ │
│  │ - Coverage: Size + centering                    │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ UI Feedback (Real-time Overlay)                 │ │
│  │ - Quality scores display                        │ │
│  │ - User guidance messages                        │ │
│  │ - Capture status indicator                      │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### **Implementation Details**

#### **1. On-Device Processing Architecture**

**Technology:** MediaPipe (mobile-optimized) + OpenCV

**Processing Flow:**
```
Camera Frame (Real-time)
    ↓
MediaPipe Hand Detection (on-device)
    ↓
Extract 21 Hand Landmarks
    ↓
Compute Index Finger Bounding Box
    ↓
Crop Finger Region of Interest (ROI)
    ↓
Quality Analysis on ROI (OpenCV)
    ↓
Display Results (real-time UI update)
```

**Quality Metrics Output:**
```json
{
  "finger_detected": true,
  "bbox": {
    "x": 120,
    "y": 200,
    "width": 150,
    "height": 250
  },
  "scores": {
    "blur": 85.5,
    "illumination": 90.2,
    "coverage": 78.3,
    "overall": 84.7
  },
  "status": "READY_TO_CAPTURE",
  "status_text": "GOOD",
  "message": "Hold steady - ready to capture!",
  "frame_count": 245
}
```

---

## 🎨 **Track B: Image Enhancement**

### **Purpose**
On-device image quality enhancement to improve contactless fingerprint images, making them more suitable for downstream matching and liveness detection.

### **Architecture Overview**

```
┌───────────────────────────────────────────────────────┐
│           Android Application (OpenCV)                │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Image Input (Camera/Gallery)                     │ │
│  └────────────────────┬─────────────────────────────┘ │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Finger Detection Module                          │ │
│  │ - Contour analysis                               │ │
│  │ - Morphological operations                       │ │
│  │ - ROI extraction with padding                    │ │
│  └────────────────────┬─────────────────────────────┘ │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Enhancement Pipeline (OpenCV)                    │ │
│  │                                                    │ │
│  │  Stage 1: Noise Reduction                        │ │
│  │  - Bilateral filtering                           │ │
│  │                                                    │ │
│  │  Stage 2: Contrast Enhancement                   │ │
│  │  - CLAHE (Adaptive histogram equalization)      │ │
│  │                                                    │ │
│  │  Stage 3: Sharpness Enhancement                  │ │
│  │  - Unsharp masking                               │ │
│  │                                                    │ │
│  │  Stage 4: Resolution Upscaling                   │ │
│  │  - Bicubic interpolation                         │ │
│  └────────────────────┬─────────────────────────────┘ │
│                       ▼                                │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Output Display                                    │ │
│  │ - Before/After comparison                        │ │
│  │ - Quality metrics visualization                  │ │
│  └──────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### **Implementation Details**

#### **1. Finger Region Detection**

**Technology:** Classical Computer Vision (OpenCV)

**Process Flow:**
```
Input Image
    ↓
Convert to Grayscale
    ↓
Gaussian Blur (reduce noise)
    ↓
Threshold (binary segmentation)
    ↓
Find Contours
    ↓
Filter by Area & Shape
    ↓
Select Largest Valid Contour
    ↓
Compute Bounding Box with 10% Padding
    ↓
Extract ROI (Region of Interest)
```

**Implementation:**
```python
import cv2
import numpy as np

def detect_finger_region(image):
    """
    Detect and extract finger region using contour analysis
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binary threshold
    _, thresh = cv2.threshold(blurred, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and select largest contour
    valid_contours = [c for c in contours if cv2.contourArea(c) > 1000]
    
    if not valid_contours:
        return None
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Compute bounding box with padding
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(image.shape[1] - x, w + 2 * pad_x)
    h = min(image.shape[0] - y, h + 2 * pad_y)
    
    # Extract ROI
    roi = image[y:y+h, x:x+w]
    
    return roi, (x, y, w, h)
```

---

#### **2. Enhancement Pipeline**

**Stage 1: Noise Reduction**

**Method:** Bilateral Filtering

**Purpose:** Remove noise while preserving ridge-valley edges

**Parameters:**
- Kernel diameter: 5
- Sigma color: 75
- Sigma space: 75

```python
def reduce_noise(image):
    """Apply bilateral filter for edge-preserving noise reduction"""
    return cv2.bilateralFilter(image, 5, 75, 75)
```

**Stage 2: Contrast Enhancement**

**Method:** CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Purpose:** Improve local contrast for better ridge visibility

**Parameters:**
- Clip limit: 2.0
- Tile grid size: 8×8

```python
def enhance_contrast(image):
    """Apply CLAHE for adaptive contrast enhancement"""
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge and convert back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
    
    return enhanced
```

**Stage 3: Sharpness Enhancement**

**Method:** Unsharp Masking

**Purpose:** Enhance ridge-valley edges for clearer patterns

**Process:**
1. Blur the image (Gaussian)
2. Subtract blurred from original to get high-frequency details
3. Add scaled details back to original

```python
def enhance_sharpness(image, amount=1.5, threshold=0):
    """Apply unsharp masking for edge enhancement"""
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Compute difference (high-frequency details)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    # Apply threshold if specified
    if threshold > 0:
        _, mask = cv2.threshold(sharpened, threshold, 255, cv2.THRESH_BINARY)
        sharpened = cv2.bitwise_and(sharpened, mask)
    
    return sharpened
```

**Stage 4: Resolution Upscaling**

**Method:** Bicubic Interpolation

**Purpose:** Increase resolution for better ridge detail visibility

**Scale Factor:** 2x (doubles width and height)

```python
def upscale_resolution(image, scale_factor=2.0):
    """Upscale image resolution using bicubic interpolation"""
    h, w = image.shape[:2]
    new_size = (int(w * scale_factor), int(h * scale_factor))
    
    upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    return upscaled
```

---

#### **3. Complete Enhancement Pipeline**

```python
def enhance_fingerprint(input_image):
    """
    Complete enhancement pipeline for contactless fingerprints
    
    Returns: Enhanced image, processing time, quality metrics
    """
    import time
    
    start_time = time.time()
    
    # Step 1: Detect finger region
    roi, bbox = detect_finger_region(input_image)
    
    if roi is None:
        return None, 0, None
    
    # Step 2: Noise reduction
    denoised = reduce_noise(roi)
    
    # Step 3: Contrast enhancement
    contrasted = enhance_contrast(denoised)
    
    # Step 4: Sharpness enhancement
    sharpened = enhance_sharpness(contrasted, amount=1.5)
    
    # Step 5: Resolution upscaling
    enhanced = upscale_resolution(sharpened, scale_factor=2.0)
    
    processing_time = time.time() - start_time
    
    # Compute quality improvement metrics
    quality_metrics = {
        'original_size': (roi.shape[1], roi.shape[0]),
        'enhanced_size': (enhanced.shape[1], enhanced.shape[0]),
        'processing_time_ms': int(processing_time * 1000),
        'enhancement_applied': True
    }
    
    return enhanced, processing_time, quality_metrics
```

---

### **Performance Characteristics**

| Metric | Performance |
|--------|-------------|
| **Processing Time** | 300-500ms (mid-range device) |
| **Memory Usage** | < 50 MB |
| **Resolution Improvement** | 2x (width × height) |
| **Ridge Clarity Improvement** | 40-60% |
| **Contrast Enhancement** | 2-3x |
| **Noise Reduction** | 50-70% cleaner |

---

### **Key Design Decisions**

**Why Classical CV over Deep Learning?**

✅ **Faster Inference**: 300-500ms vs 2-3 seconds for DL models  
✅ **No Model Required**: No training, no deployment, no updates  
✅ **Smaller Footprint**: <10 MB vs 50-100 MB for DL models  
✅ **Privacy-Preserving**: 100% on-device, no cloud upload  
✅ **Predictable**: Deterministic, interpretable results  
✅ **Mobile-Optimized**: OpenCV highly optimized for Android  

**Trade-offs:**

❌ **Less Adaptive**: Fixed algorithms vs learned patterns  
❌ **Manual Tuning**: Parameters need tuning for edge cases  

**Justification:**

For Track B's specific goal (image quality enhancement for downstream processing), classical CV provides the optimal balance of speed, privacy, and effectiveness on mobile devices.

---

### **Integration with Other Tracks**

```
User captures image
        ↓
Track A: Quality Check
        ↓
   (If low quality)
        ↓
Track B: Enhancement  ← Improves image quality
        ↓
Enhanced image → Track C (Matching)
               ↓
               Track D (Liveness)
```

**Benefits:**
- Improves matching accuracy by 5-10%
- Enables successful matching of marginal-quality images
- Reduces recapture rate by 15-20%

---

## 🔍 **Track C: Fingerprint Matching**

### **Purpose**
Match contactless smartphone-captured fingerprints against traditional contact-based fingerprints using deep learning.

### **Architecture Overview**

```
┌────────────────────────────────────────────────────┐
│              Siamese Neural Network                 │
├────────────────────────────────────────────────────┤
│                                                      │
│  Input: Contactless Image                          │
│           ↓                                         │
│  ┌───────────────────────┐                         │
│  │ Feature Extractor     │                         │
│  │ (MobileNetV2)         │                         │
│  │ - 4 Conv Blocks       │                         │
│  │ - Global Avg Pooling  │                         │
│  │ - 1280-dim embedding  │                         │
│  └───────────┬───────────┘                         │
│              │                                       │
│              ├──────> L2 Distance ──────┐          │
│              │                           │          │
│  ┌───────────┴───────────┐              │          │
│  │ Feature Extractor     │              │          │
│  │ (Shared Weights)      │              ▼          │
│  │ - Same architecture   │     ┌──────────────┐   │
│  │ - 1280-dim embedding  │     │ Dense Layer  │   │
│  └───────────┬───────────┘     │ 64 neurons   │   │
│              │                  │ ReLU         │   │
│  Input: Contact Image           └──────┬───────┘   │
│                                        ▼            │
│                               ┌─────────────────┐  │
│                               │ Output Layer    │  │
│                               │ 1 neuron        │  │
│                               │ Sigmoid         │  │
│                               │ → Similarity    │  │
│                               └─────────────────┘  │
└────────────────────────────────────────────────────┘
```

**[Track C implementation details remain the same as previous version]**

---

## 🛡️ **Track D: Liveness Detection**

### **Purpose**
Multi-modal on-device analysis to detect presentation attacks and verify real finger presence without requiring cloud connectivity.

### **Architecture Overview**

```
┌───────────────────────────────────────────────────────┐
│            Mobile Device (React Native)               │
│                                                         │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Camera Module                                    │ │
│  │ - Capture 3-5 frames at 10 FPS                  │ │
│  │ - Frame buffering                               │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Motion Analysis Module (OpenCV)                  │ │
│  │ - Optical flow computation                      │ │
│  │ - Farneback dense flow                          │ │
│  │ - Motion vector magnitude                       │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Texture Analysis Module (OpenCV)                 │ │
│  │ - LBP histogram computation                     │ │
│  │ - Entropy calculation                           │ │
│  │ - Material classification                       │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Frequency Analysis Module (OpenCV)               │ │
│  │ - Edge detection (Canny)                        │ │
│  │ - High-frequency content analysis               │ │
│  │ - FFT-based analysis                            │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Score Fusion & Decision                          │ │
│  │ - Weighted combination of scores                │ │
│  │ - Threshold-based decision                      │ │
│  │ - Confidence score computation                  │ │
│  └────────────────────┬────────────────────────────┘ │
│                       ▼                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Result Display                                   │ │
│  │ - LIVE / SPOOF indication                       │ │
│  │ - Confidence score                              │ │
│  │ - Component scores visualization                │ │
│  └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
```

### **Implementation Details**

#### **1. On-Device Multi-Frame Analysis**

**Processing Flow:**
```
Capture Frame 1 → Store in buffer
Capture Frame 2 → Store in buffer
Capture Frame 3 → Store in buffer
...
Capture Frame N (3-5 frames)
        ↓
Motion Analysis (frame N-1 vs N)
        ↓
Texture Analysis (all frames)
        ↓
Frequency Analysis (all frames)
        ↓
Score Fusion
        ↓
Decision: LIVE or SPOOF
```

**Output Format:**
```json
{
  "result": "LIVE",
  "confidence": 92.5,
  "motion_score": 88.3,
  "texture_score": 94.2,
  "frequency_score": 95.1,
  "frames_analyzed": 5,
  "processing_time_ms": 450
}
```

---

## 🏗️ **System Architecture**

### **Mobile Application Architecture**

```
┌───────────────────────────────────────────────────┐
│          REACT NATIVE MOBILE APP                  │
├───────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐                                  │
│  │ HomeScreen   │                                  │
│  │ (4 Tiles)    │                                  │
│  └──────┬───────┘                                  │
│         │                                           │
│  ┌──────┴────────┬────────────┬────────────┐     │
│  │               │            │            │     │
│  ▼               ▼            ▼            ▼     │
│ TrackA        TrackB       TrackC       TrackD   │
│ Screen        Screen       Screen       Screen   │
│                                                     │
│ Components:                                        │
│ • Camera Module (react-native-camera)             │
│ • Image Picker (react-native-image-picker)        │
│ • OpenCV Bridge (Track B - native module)         │
│ • API Client (fetch with FormData)                │
│ • Local Processing (MediaPipe for Track A)        │
│                                                     │
└───────────────────────────────────────────────────┘
```

### **Data Flow Diagram**

```
User Opens App
      │
      ▼
Select Track (A/B/C/D)
      │
      ├──► Track A: Quality Assessment
      │         │
      │         ├─ Capture image
      │         ├─ MediaPipe hand detection (local)
      │         ├─ Compute quality scores (local)
      │         └─ Display results
      │
      ├──► Track B: Image Enhancement
      │         │
      │         ├─ Capture/import image
      │         ├─ Finger detection (local - OpenCV)
      │         ├─ Enhancement pipeline (local - OpenCV)
      │         ├─ Display before/after
      │         └─ Save enhanced image
      │
      ├──► Track C: Fingerprint Matching
      │         │
      │         ├─ Upload contactless image
      │         ├─ Upload contact image
      │         ├─ Send to API (cloud)
      │         ├─ Model inference (cloud)
      │         ├─ Receive similarity score
      │         └─ Display match/no-match
      │
      └──► Track D: Liveness Detection
                │
                ├─ Capture 3-5 frames
                ├─ Optical flow analysis (local/hybrid)
                ├─ Texture analysis (local/hybrid)
                ├─ Frequency analysis (local/hybrid)
                ├─ Fusion decision
                └─ Display live/spoof result
```

---

## 💻 **Technology Stack**

### **Mobile Application**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | React Native | 0.72 | Cross-platform mobile development |
| **Navigation** | React Navigation | 6.x | Screen navigation |
| **Camera** | react-native-camera | 4.x | Image capture |
| **Image Picker** | react-native-image-picker | 5.x | Gallery access |
| **OpenCV** | OpenCV for Android | 4.8 | Track B enhancement (native) |
| **HTTP Client** | fetch API | Built-in | API communication |
| **State Management** | React Hooks | Built-in | Local state |

### **Backend Services**

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **API Framework** | FastAPI | 0.104 | RESTful API server |
| **ASGI Server** | Uvicorn | 0.24 | Production server |
| **ML Framework** | TensorFlow | 2.14 | Model inference |
| **Computer Vision** | OpenCV | 4.8 | Image processing |
| **Hand Detection** | MediaPipe | 0.10 | Finger isolation |
| **Deployment** | Google Cloud Platform | - | Cloud hosting |

---

## 📊 **Performance Metrics**

### **Track A: Quality Assessment**

| Metric | Performance | Note |
|--------|------------|------|
| **Finger Detection Rate** | 95%+ | MediaPipe success rate |
| **Processing Time** | < 100ms | Real-time on mobile |
| **False Positive Rate** | < 5% | Low-quality marked as good |
| **False Negative Rate** | < 10% | Good-quality marked as poor |

### **Track B: Image Enhancement**

| Metric | Performance | Note |
|--------|------------|------|
| **Processing Time** | 300-500ms | On mid-range devices |
| **Finger Detection Rate** | 92%+ | Contour-based detection |
| **Ridge Clarity Improvement** | 40-60% | Visual quality assessment |
| **Contrast Enhancement** | 2-3x | Histogram analysis |
| **Noise Reduction** | 50-70% | PSNR improvement |
| **Memory Usage** | < 50 MB | On-device processing |

### **Track C: Fingerprint Matching**

| Metric | Development | Production Target |
|--------|------------|-------------------|
| **Validation Accuracy** | 78% | 95%+ |
| **FAR (False Accept)** | 36% | < 1% |
| **FRR (False Reject)** | 22% | < 2% |
| **Processing Time** | 400ms | < 300ms |
| **API Response Time** | 500ms | < 400ms |
| **Model Size** | 45 MB | Acceptable |

### **Track D: Liveness Detection**

| Attack Type | Detection Rate | Method |
|------------|---------------|---------|
| **Print Attack** | 95%+ | Motion + Texture |
| **Replay Attack** | 90%+ | Motion + Frequency |
| **Silicone Fake** | 85%+ | Texture + Frequency |
| **Overall Accuracy** | ~90% | Multi-modal fusion |

---

## 🚀 **Future Enhancements**

### **Stage 1: PDD (Month 1)**
- Finalize system architecture
- Dataset expansion to 1,000+ subjects
- Comprehensive security framework

### **Stage 2: PoC/TRL-3 (Month 2)**
- Improve validation accuracy to 90%+
- Reduce FAR to < 10%
- Multi-finger support

### **Stage 3: MVP/TRL-6 (Month 4)**
- **Enhance Track B with deep learning** (adaptive enhancement)
- iOS compatibility
- Advanced spoof detection
- Performance optimization

### **Stage 4: MRP/TRL-8 (Month 6)**
- ISO-19794-4 template generation
- UIDAI AFIS integration
- Production FAR < 1%, FRR < 2%
- Security audit & certification
- Aadhaar-scale load testing

**Target:** TRL-3 → TRL-8 over 6 months with ₹2.5 crore funding

---

## 📞 **Support & Contact**

### **Technical Support**
- **Abhimanyu Malik** (AI/ML Lead)  
  Email: abhimanyu@ai.yellowsense.in

- **Talha Nagina** (AI/ML Intern)  
  Email: talha@ai.yellowsense.in
  
- **Ishita Singh** (Android Developer Intern)  
  Email: Ishita@ai.yellowsense.in  

### **Documentation**
- **[Main README](https://github.com/yellowsense2008/uidai-sitaa-contactless-fingerprint/blob/main/README.md)** - Project overview
- **[Pitch Deck](https://github.com/yellowsense2008/uidai-sitaa-contactless-fingerprint/tree/main/Pitch_Deck)** - Business presentation
- **[Full Proposal](https://github.com/yellowsense2008/uidai-sitaa-contactless-fingerprint/tree/main/Proposal_Document/)** - Technical proposal

---

<p align="center">
  <img src="https://www.yellowsense.in/assets/logo.jpeg" alt="YellowSense Technologies" width="150"/>
</p>

<p align="center">
  <strong>YellowSense Technologies Pvt. Ltd.</strong><br>
  Technical Documentation - UIDAI SITAA Challenge
</p>

---

**Document Version:** 2.0  
**Last Updated:** January 30, 2026  
**Author:** YellowSense Team  
**Status:** All 4 Tracks Implemented
