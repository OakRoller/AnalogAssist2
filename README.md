# GoFundMe: https://www.gofundme.com/f/help-launch-analogassist2-a-smart-light-meter-for-film

# 📸 AnalogAssist2

**AnalogAssist2** is an experimental camera assistant app built for iOS and Apple Watch that combines **real-time computer vision**, **Core ML segmentation**, and **classic exposure metering** principles to help photographers make intelligent, scene-aware exposure decisions — just like using a handheld light meter for analog cameras.

---

## 🚀 Features

### 🎥 Real-time Camera + CoreML
- Uses **DETR ResNet50 Semantic Segmentation (F16P8)** model for real-time scene analysis.
- Performs **zonal**, **scene-based**, and **subject-biased** metering using 18% gray reference.
- Live camera overlay with segmentation color mask and labeled object areas.
- Automatically identifies the **main subject** via saliency and geometry heuristics.

### ⌚ Apple Watch Integration
- Companion watchOS app for remote control and ISO adjustments.
- Displays **Zonal**, **Scene**, and **Main-biased** exposure suggestions in real time.
- ISO can be adjusted using **Digital Crown** or buttons and syncs instantly with iPhone.
- Bidirectional communication using **WatchConnectivity (WCSession)**:
  - iPhone → Watch: live exposure & ISO updates.
  - Watch → iPhone: ISO change events.

### ⚙️ Intelligent Metering Logic
- Three metering modes:
  - **Zonal Metering** – 6×6 grid, 18% gray average.
  - **Scene Equivalent Metering** – based on device exposure data.
  - **Main Subject Metering** – applies bias depending on detected subject area and luminance.
- 18% gray normalization ensures consistent exposure simulation for film-style workflows.

---

## 🧩 Technical Details

- **Languages:** Swift 5.10, SwiftUI, Combine  
- **Frameworks:** AVFoundation, Vision, CoreML, WatchConnectivity  
- **Devices:** iPhone + Apple Watch (paired)  
- **Model:** `DETRResNet50SemanticSegmentationF16P8.mlpackage`  
- **Minimum iOS version:** iOS 17  
- **Minimum watchOS version:** watchOS 10  

---

## 🖼️ App Icon & Assets

Each target has its own icon set:
- `AnalogAssist2/Assets.xcassets` → iOS app icons  
- `AnalogAssist2 Watch App/Assets.xcassets` → watchOS icons  

To regenerate:
1. Add a 1024×1024 PNG (square, no rounded corners).  
2. Right-click → *Generate App Icons for all sizes* (Xcode 15+).  
3. Confirm in **Target → General → App Icon** that the name matches.

---

## 🔗 Communication Flow

```text
iPhone (Camera + Model)
        ↓ pushMeters()
 WatchConnectivity → sendMessage / updateApplicationContext
        ↓
Apple Watch (WatchSession)
        ↳ applyContext() updates UI + ISO display
        ↑
Digital Crown → sendISO() → WCSession.sendMessage → iPhone
```

---

## 🧰 Build & Run

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/AnalogAssist2.git
   cd AnalogAssist2
   ```
2. Open `AnalogAssist2.xcodeproj` in **Xcode 15 or later**.
3. Set the **Team** and **Bundle Identifier** for both targets:
   - `AnalogAssist2` (iOS)
   - `AnalogAssist2 Watch App` (watchOS)
4. Run the **iOS App** on a real iPhone (Camera required).
5. Install the **Watch App** through Xcode or automatically from the iPhone app.

---

## 🧪 Debugging Tips

- **If Watch shows “Waiting for iPhone…”**
  - Ensure both apps are running and paired.
  - Check Xcode console for `[WC watch]` and `[WC iOS]` logs.
- **If ISO reverts or doesn’t sync:**
  - Confirm both sides’ `WCSession.default.isReachable` are `true`.
  - Ensure `applyContext()` updates happen on the main thread.
- **For development:** enable debug logs in `PhoneWatchConnectivity` and `WatchSession`.

---

## 🧑‍💻 Author

**Yinuo Wang**  
📸 Analog photography enthusiast · 💻 Developer

---

## 📜 License

MIT License © 2025 Yinuo Wang
