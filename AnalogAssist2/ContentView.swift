// ContentView.swift — iOS side with WatchConnectivity
// AnalogAssist2
//
// iPhone camera + segmentation + three metering modes (all 18% gray)
// + WatchConnectivity: push meters/ISO to Apple Watch & receive ISO changes
//

import SwiftUI
import AVFoundation
import Vision
import CoreML
import CoreImage
import Accelerate
import ImageIO
import WatchConnectivity
import Foundation

// MARK: - Logging

private func log(_ msg: @autoclosure () -> String) {
    #if DEBUG
    let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
    print("[SegCam \(ts)] \(msg())")
    #endif
}

// MARK: - Core ML raw metadata keys + safe accessor

private enum CoreMLMeta {
    static let description    = "com.apple.coreml.model.description"
    static let author         = "com.apple.coreml.model.author"
    static let version        = "com.apple.coreml.model.version"
    static let creatorDefined = "com.apple.coreml.model.creator_defined"
}

@inline(__always)
private func metaString(_ md: MLModelDescription, _ key: String) -> String? {
    (md.metadata as NSDictionary)[key] as? String
}

// Clean up labels for display (remove quotes/backticks/whitespace)
@inline(__always)
private func cleanLabel(_ s: String) -> String {
    s.trimmingCharacters(in: CharacterSet(charactersIn: " \t\n\r\"'`"))
}

// MARK: - sRGB → Linear helpers

@inline(__always)
private func srgbToLinear01(_ x255: Double) -> Double {
    let x = max(0.0, min(1.0, x255 / 255.0))
    if x <= 0.04045 { return x / 12.92 }
    return pow((x + 0.055)/1.055, 2.4)
}

@inline(__always)
private func linearLumaFromBGRA(_ b: Double, _ g: Double, _ r: Double) -> Double {
    // Convert sRGB bytes to linear, then 0.2126/0.7152/0.0722
    let rl = srgbToLinear01(r)
    let gl = srgbToLinear01(g)
    let bl = srgbToLinear01(b)
    return 0.2126*rl + 0.7152*gl + 0.0722*bl // 0..1 linear
}

// MARK: - Bundle + model helpers

func loadModelFromBundle(name: String, configuration cfg: MLModelConfiguration) throws -> MLModel {
    #if SWIFT_PACKAGE
    let bundle = Bundle.module
    #else
    let bundle = Bundle.main
    #endif

    if let url = bundle.url(forResource: name, withExtension: "mlmodelc") {
        log("Loading compiled model: \(url.lastPathComponent)")
        return try MLModel(contentsOf: url, configuration: cfg)
    }
    if let pkg = bundle.url(forResource: name, withExtension: "mlpackage") {
        log("Compiling .mlpackage at runtime…")
        let compiled = try MLModel.compileModel(at: pkg)
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }
    if let raw = bundle.url(forResource: name, withExtension: "mlmodel") {
        log("Compiling .mlmodel at runtime…")
        let compiled = try MLModel.compileModel(at: raw)
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }
    if let anyC = bundle.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)?.first {
        log("Auto-discovered compiled model: \(anyC.lastPathComponent)")
        return try MLModel(contentsOf: anyC, configuration: cfg)
    }
    if let anyP = bundle.urls(forResourcesWithExtension: "mlpackage", subdirectory: nil)?.first {
        log("Auto-discovered .mlpackage: \(anyP.lastPathComponent) (compiling)")
        let compiled = try MLModel.compileModel(at: anyP)
        return try MLModel(contentsOf: compiled, configuration: cfg)
    }
    throw NSError(domain: "CoreML", code: -1,
                  userInfo: [NSLocalizedDescriptionKey: "No .mlmodelc/.mlpackage/.mlmodel found in bundle. Check Target Membership & Copy Bundle Resources."])
}

// MARK: - Labels: parse + discover from metadata/files

private func parseLabels(from string: String) -> [String] {
    let s = string.trimmingCharacters(in: .whitespacesAndNewlines)
    if s.first == "[", s.last == "]",
       let data = s.data(using: .utf8),
       let arr = (try? JSONSerialization.jsonObject(with: data)) as? [Any] {
        return arr.compactMap { $0 as? String }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
    let separators = CharacterSet(charactersIn: ",;\n\t\r")
    return s.components(separatedBy: separators)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
}

private func looksLikeLabels(_ arr: [String]) -> Bool {
    guard arr.count >= 2 else { return false }
    if arr.count > 4096 { return false }
    let shortish = arr.filter { $0.count <= 32 }.count
    return shortish >= max(2, arr.count / 4)
}

private func findLabelsRecursively(in dict: [String: Any], path: String = "root") -> ([String], String)? {
    let candidateKeys = Set(["labels","classes","classLabels","labelNames","categories","class_names","names","labels_csv"])
    for (k, v) in dict {
        if let arr = v as? [String], looksLikeLabels(arr) { return (arr, "\(path).\(k)") }
    }
    for (k, v) in dict {
        if let s = v as? String {
            let arr = parseLabels(from: s)
            if looksLikeLabels(arr) { return (arr, "\(path).\(k)") }
        }
    }
    for (k, v) in dict {
        if let sub = v as? [String: Any], candidateKeys.contains(k) {
            if let (labels, p) = findLabelsRecursively(in: sub, path: "\(path).\(k)") { return (labels, p) }
        }
    }
    for (k, v) in dict {
        if let sub = v as? [String: Any] {
            if let (labels, p) = findLabelsRecursively(in: sub, path: "\(path).\(k)") { return (labels, p) }
        } else if let arrOfDicts = v as? [[String: Any]] {
            for (i, sub) in arrOfDicts.enumerated() {
                if let (labels, p) = findLabelsRecursively(in: sub, path: "\(path).\(k)[\(i)]") { return (labels, p) }
            }
        }
    }
    return nil
}

func readSegmentationLabelsFromMetadata(_ model: MLModel) -> [String]? {
    let md = model.modelDescription
    let nsMeta = md.metadata as NSDictionary
    var flat: [String: Any] = [:]
    if let s = nsMeta[CoreMLMeta.description] as? String { flat["description"] = s }
    if let s = nsMeta[CoreMLMeta.author] as? String { flat["author"] = s }
    if let s = nsMeta[CoreMLMeta.version] as? String { flat["version"] = s }
    if let cd = nsMeta[CoreMLMeta.creatorDefined] as? [String: Any] { flat["creator_defined"] = cd }
    for (kAny, vAny) in md.metadata {
        let key = String(describing: kAny)
        flat["meta.\(key)"] = vAny
    }
    if let (labels, whereFound) = findLabelsRecursively(in: flat, path: "meta") {
        log("Loaded \(labels.count) labels from metadata at \(whereFound)")
        return labels
    }
    log("No labels found in metadata (scanned recursively).")
    return nil
}

func readLabelsFromBundleFiles() -> [String]? {
    #if SWIFT_PACKAGE
    let bundle = Bundle.module
    #else
    let bundle = Bundle.main
    #endif

    if let url = bundle.url(forResource: "labels", withExtension: "txt"),
       let text = try? String(contentsOf: url, encoding: .utf8) {
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !lines.isEmpty { log("Loaded \(lines.count) labels from labels.txt"); return lines }
    }
    if let url = bundle.url(forResource: "labels", withExtension: "csv"),
       let text = try? String(contentsOf: url, encoding: .utf8) {
        let first = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        let items = parseLabels(from: first)
        if !items.isEmpty { log("Loaded \(items.count) labels from labels.csv"); return items }
    }
    if let url = bundle.url(forResource: "labels", withExtension: "json"),
       let data = try? Data(contentsOf: url),
       let arr = try? JSONSerialization.jsonObject(with: data) as? [Any] {
        let items = arr.compactMap { $0 as? String }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if !items.isEmpty { log("Loaded \(items.count) labels from labels.json"); return items }
    }
    return nil
}

/// Ensure we have at least `minCount` labels by padding with class_#.
func padLabels(_ labels: [String], toAtLeast minCount: Int) -> [String] {
    if labels.count >= minCount { return labels }
    var out = labels
    out.reserveCapacity(minCount)
    for i in labels.count..<minCount { out.append("class_\(i)") }
    return out
}

// MARK: - Color (HUD & overlay unified)

@inline(__always)
private func classRGBA(_ cls: Int, boostMain: Bool) -> (r: Double, g: Double, b: Double, a: Double) {
    let a: Double = boostMain ? 0xE0 : 0x60
    let h = UInt32(cls & 255)
    let r = Double((h &* 97)  & 255)
    let g = Double((h &* 57)  & 255)
    let b = Double((h &* 157) & 255)
    return (r/255.0, g/255.0, b/255.0, a/255.0)
}

@inline(__always)
private func packBGRApremul(_ r: Double, _ g: Double, _ b: Double, _ a: Double) -> UInt32 {
    let A = UInt8(clamping: Int(a * 255.0 + 0.5))
    let R = UInt8(clamping: Int(r * a * 255.0 + 0.5))
    let G = UInt8(clamping: Int(g * a * 255.0 + 0.5))
    let B = UInt8(clamping: Int(b * a * 255.0 + 0.5))
    return (UInt32(B)) | (UInt32(G) << 8) | (UInt32(R) << 16) | (UInt32(A) << 24)
}

@inline(__always)
private func pixelBGRA(for cls: Int, boostMain: Bool) -> UInt32 {
    let c = classRGBA(cls, boostMain: boostMain)
    return packBGRApremul(c.r, c.g, c.b, c.a)
}

@inline(__always)
private func colorForClassSwiftUI(_ cls: Int, boost: Bool) -> Color {
    let c = classRGBA(cls, boostMain: boost)
    return Color(.sRGB, red: c.r, green: c.g, blue: c.b, opacity: c.a)
}

// MARK: - WatchConnectivity (iPhone side)

extension Notification.Name { static let ISOChangedFromWatch = Notification.Name("ISOChangedFromWatch") }

final class PhoneWatchConnectivity: NSObject, WCSessionDelegate {
    static let shared = PhoneWatchConnectivity()
    private override init() { super.init() }

    // === Public API ===
    private var onISOChange: ((Double) -> Void)?

    /// Call once at app launch (e.g., in @main App.init or very early on)
    func start(onISOChange: @escaping (Double) -> Void) {
        self.onISOChange = onISOChange
        guard WCSession.isSupported() else { print("[WC iOS] not supported"); return }
        let s = WCSession.default
        s.delegate = self
        print("[WC iOS] start(): delegate set? \(s.delegate === self)")
        s.activate()
    }

    /// Call whenever you want to push the latest meters/ISO to the watch
    func pushMeters(zonal: String, scene: String, subject: String, iso: Double) {
        // Keep a snapshot for background delivery
        lastContext = [
            "meterZonal": zonal,
            "meterScene": scene,
            "meterSubject": subject,
            "iso": iso
        ]
        sendLiveIfReachableAndNotEcho()
        sendContextSnapshot() // background-safe
    }

    // === Internals ===
    private var activation: WCSessionActivationState = .notActivated
    private var isUpdatingFromWatch = false // Option 2: source guard

    // last-known snapshot for updateApplicationContext
    private var lastContext: [String: Any] = [
        "meterZonal": "—",
        "meterScene": "—",
        "meterSubject": "—",
        "iso": 100.0
    ]

    // MARK: WCSessionDelegate

    func session(_ session: WCSession,
                 activationDidCompleteWith activationState: WCSessionActivationState,
                 error: Error?) {
        activation = activationState
        if let e = error { print("[WC iOS] activate error: \(e.localizedDescription)") }
        print("[WC iOS] activated state=\(activationState.rawValue), reachable=\(session.isReachable), installed=\(session.isWatchAppInstalled)")

        guard session.isWatchAppInstalled else {
            print("[WC iOS] watch app not installed; will skip sends until installed")
            return
        }

        // Push the current snapshot so watch has immediate data
        sendContextSnapshot()
    }

    #if os(iOS)
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) {
        print("[WC iOS] didDeactivate → re-activate")
        WCSession.default.activate()
    }
    func sessionReachabilityDidChange(_ session: WCSession) {
        print("[WC iOS] reachable=\(session.isReachable)")
    }
    #endif

    // Watch → iPhone (live)
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        print("[iOS] didReceiveMessage: \(message)")
        if let iso = message["iso"] as? Double {
            applyISOFromWatch(iso)
        }
    }

    // Watch → iPhone (queued)
    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String : Any]) {
        print("[iOS] didReceiveUserInfo: \(userInfo)")
        if let iso = userInfo["iso"] as? Double {
            applyISOFromWatch(iso)
        }
    }

    private func applyISOFromWatch(_ iso: Double) {
        DispatchQueue.main.async {
            self.isUpdatingFromWatch = true
            self.onISOChange?(iso) // handoff into the VM callback
            NotificationCenter.default.post(name: .ISOChangedFromWatch, object: nil, userInfo: ["iso": iso])
            self.isUpdatingFromWatch = false
        }
    }

    // MARK: Helpers

    private func sendLiveIfReachableAndNotEcho() {
        guard activation == .activated else { return }
        let s = WCSession.default
        guard s.isWatchAppInstalled else { return }
        guard s.isReachable else { return }          // live message only when reachable
        guard !isUpdatingFromWatch else { return }   // Option 2: prevent echo

        s.sendMessage(lastContext, replyHandler: nil) { err in
            print("[WC iOS] sendMessage error: \(err.localizedDescription)")
        }
    }

    private func sendContextSnapshot() {
        guard activation == .activated else { return }
        let s = WCSession.default
        guard s.isWatchAppInstalled else { return }
        do {
            try s.updateApplicationContext(lastContext) // background-safe; no reachability required
        } catch {
            print("[WC iOS] updateApplicationContext error: \(error.localizedDescription)")
        }
    }
}

// MARK: - ViewModel

final class ViewModel: NSObject, ObservableObject {
    // UI state
    @Published var overlayImage: CGImage?
    @Published var classPercentages: [(label: String, percent: Double, cls: Int)] = []
    @Published var errorMessage: String?
    @Published var isAuthorized: Bool = false
    @Published var lastUpdateText: String = ""
    @Published var mainSubjectText: String = "Main: —"
    @Published var mainClassIndex: Int = -1

    // centerCrop meta-rect
    @Published var centerCropMetaRect: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)

    // Watch sync kick
    func pushToWatch() {
        PhoneWatchConnectivity.shared.pushMeters(
            zonal: meterZonal,
            scene: meterMain,
            subject: meterSubject,
            iso: userISO
        )
    }

    // Camera
    let session = AVCaptureSession()
    private var request: VNCoreMLRequest?
    private var activeDevice: AVCaptureDevice?

    // Labels
    private var labels: [String] = []

    // Background inference
    private let inferQueue = DispatchQueue(label: "seg.infer.queue", qos: .userInitiated)
    private let frameGate = DispatchSemaphore(value: 1)

    // Accuracy/time knobs
    private let computeUnits: MLComputeUnits = .all
    private let frameSkip: Int = 0
    private var frameCounter: Int = 0
    private let useFlipTTA: Bool = true
    private let useModeFilter3x3: Bool = true

    // Metering IO
    @Published var userISO: Double = 100
    @Published var meterZonal: String = "—"
    @Published var meterMain: String = "—"
    @Published var meterAlternates: [String] = []
    @Published var meterSubject: String = "—"

    private let targetShutterForApertureSuggestions: Double = 1.0 / 60.0
    private let fullStops: [Double] = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]

    private var lastEV100: Double = .nan
    private var lastApertureN: Double = 1.8

    // Subject-biased metering (all 18% gray)
    private let targetGray: Double = 0.18
    private let biasExponent: Double = 1.0
    private let biasMax: Double = 1.0
    private let useClipGuards: Bool = true
    private let clipLow: Double = 0.02
    private let clipHigh: Double = 0.98

    // Zonal metering (all 18% gray; no center emphasis)
    private let zoneCols: Int = 6
    private let zoneRows: Int = 6
    private let zoneTargetGray: Double = 0.18
    private let zoneClipLow: Double = 0.02
    private let zoneClipHigh: Double = 0.98

    // MARK: Model & labels loading

    func loadModel() async {
        do {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = computeUnits
            let model = try loadModelFromBundle(name: "DETRResnet50SemanticSegmentationF16P8", configuration: cfg)

            let md = model.modelDescription
            let prettyName =
                metaString(md, CoreMLMeta.description) ??
                metaString(md, CoreMLMeta.author) ??
                metaString(md, CoreMLMeta.version) ??
                "Core ML model"
            log("Model loaded: \(prettyName)")
            log("Inputs: \(md.inputDescriptionsByName.keys.sorted())")
            log("Outputs: \(md.outputDescriptionsByName.keys.sorted())")
            if let out = md.outputDescriptionsByName.values.first { log("First output type: \(out.type)") }

            let fileLabels = readLabelsFromBundleFiles()
            let metaLabels = fileLabels ?? readSegmentationLabelsFromMetadata(model)
            let base = metaLabels ?? []
            self.labels = padLabels(base, toAtLeast: 1000)
            log("Active labels count: \(self.labels.count)")

            didLoadModel(model)
        } catch {
            let msg = "The model failed to load: \(error.localizedDescription)"
            log(msg)
            DispatchQueue.main.async { [weak self] in self?.errorMessage = msg }
        }
    }

    private func didLoadModel(_ model: MLModel) {
        do {
            let vnModel = try VNCoreMLModel(for: model)
            let req = VNCoreMLRequest(model: vnModel)
            req.imageCropAndScaleOption = .centerCrop
            self.request = req
            log("Vision request prepared (imageCropAndScaleOption = .centerCrop)")
        } catch {
            let msg = "Failed to prepare Vision model: \(error.localizedDescription)"
            log(msg)
            DispatchQueue.main.async { [weak self] in self?.errorMessage = msg }
        }
    }

    // MARK: Camera

    func requestAuthorizationAndStart() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.isAuthorized = true }
            startSession()
        case .notDetermined:
            log("Requesting camera access…")
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                log("Camera access: \(granted)")
                DispatchQueue.main.async { self.isAuthorized = granted }
                if granted { self.startSession() }
            }
        default:
            DispatchQueue.main.async {
                self.isAuthorized = false
                self.errorMessage = "Camera permission denied."
            }
        }
    }

    private func startSession() {
        guard session.inputs.isEmpty, session.outputs.isEmpty else {
            log("Session already configured, starting…")
            session.startRunning()
            return
        }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        do {
            guard let device = AVCaptureDevice.default(for: .video) else {
                let msg = "No camera device available"
                log(msg); DispatchQueue.main.async { self.errorMessage = msg }
                session.commitConfiguration(); return
            }
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }

            let out = AVCaptureVideoDataOutput()
            out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            out.alwaysDiscardsLateVideoFrames = true
            out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "seg.capture.queue", qos: .userInteractive))
            if session.canAddOutput(out) { session.addOutput(out) }

            self.activeDevice = device

            log("Camera configured. Preset: \(session.sessionPreset.rawValue)")
        } catch {
            let msg = "Camera setup failed: \(error.localizedDescription)"
            log(msg); DispatchQueue.main.async { self.errorMessage = msg }
        }

        session.commitConfiguration()
        session.startRunning()
        log("Session started.")
    }

    // MARK: Inference (+ saliency, TTA, filtering)

    private func exifFrom(angleDegrees: Double, mirrored: Bool) -> CGImagePropertyOrientation {
        let a = Int(((angleDegrees.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360)).rounded())
        let k = ((a + 45) / 90) * 90 % 360
        switch (k, mirrored) {
        case (0, false):   return .up
        case (0, true):    return .upMirrored
        case (90, false):  return .right
        case (90, true):   return .rightMirrored
        case (180, false): return .down
        case (180, true):  return .downMirrored
        case (270, false): return .left
        case (270, true):  return .leftMirrored
        default:           return .up
        }
    }

    private func runInference(on pixelBuffer: CVPixelBuffer,
                              exif: CGImagePropertyOrientation) {
        guard let request = self.request else {
            log("Vision request is nil")
            frameGate.signal()
            return
        }

        inferQueue.async { [weak self] in
            guard let self else { return }

            func performOnce(orientation: CGImagePropertyOrientation)
            -> (overlay: CGImage, percentages: [(String, Double, Int)], mainName: String, mainIdx: Int, indexMap: [Int], W: Int, H: Int)? {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
                let saliencyReq = VNGenerateAttentionBasedSaliencyImageRequest()
                do {
                    try handler.perform([request, saliencyReq])
                    let results = request.results ?? []
                    var saliencyPB: CVPixelBuffer?
                    if let salObs = saliencyReq.results?.first as? VNSaliencyImageObservation {
                        saliencyPB = salObs.pixelBuffer
                    }

                    if let obs = results.first as? VNCoreMLFeatureValueObservation {
                        if let r = self.postprocessFeatureValue(obs, saliencyPB: saliencyPB) {
                            return (r.overlay, r.percentages, r.mainName, r.mainIdx, r.indexMap, r.W, r.H)
                        }
                    } else if let pix = results.first as? VNPixelBufferObservation {
                        if let r = self.postprocessClassIndexMap(pix.pixelBuffer, saliencyPB: saliencyPB) {
                            return (r.overlay, r.percentages, r.mainName, r.mainIdx, r.indexMap, r.W, r.H)
                        }
                    }
                    return nil
                } catch {
                    self.logAndSetError("Vision request failed: \(error.localizedDescription)")
                    return nil
                }
            }

            // 1) 原始推理
            guard let pass1 = performOnce(orientation: exif) else {
                DispatchQueue.main.async { self.frameGate.signal() }
                return
            }

            var fusedIndex = pass1.indexMap
            let W = pass1.W, H = pass1.H
            let mainIdx = pass1.mainIdx
            let mainName = pass1.mainName

            // 2) flip-TTA（可选）
            if self.useFlipTTA {
                let exifFlip: CGImagePropertyOrientation = {
                    switch exif {
                    case .up: return .upMirrored
                    case .down: return .downMirrored
                    case .left: return .leftMirrored
                    case .right: return .rightMirrored
                    case .upMirrored: return .up
                    case .downMirrored: return .down
                    case .leftMirrored: return .left
                    case .rightMirrored: return .right
                    @unknown default: return exif
                    }
                }()
                if let pass2 = performOnce(orientation: exifFlip) {
                    var r2Index = self.flipIndexMapHoriz(pass2.indexMap, W: W, H: H)
                    for i in 0..<fusedIndex.count {
                        fusedIndex[i] = self.majority(fusedIndex[i], r2Index[i])
                    }
                }
            }

            // 3) 3×3 模式滤波（可选）
            if self.useModeFilter3x3 {
                self.mode3x3(&fusedIndex, W: W, H: H)
            }

            // 4) 由融合后的 index map 生成最终 overlay + 百分比
            let (finalImg, finalPct, counts) = self.overlayFromIndexMap(indexMap: fusedIndex, W: W, H: H, mainIdx: mainIdx)

            // 5) 主体偏置测光（B，18% 灰 + 面积权重）
            if !self.lastEV100.isNaN {
                if let (deltaEV, w) = self.subjectBiasedDeltaEV(from: pixelBuffer,
                                                                mask: fusedIndex, W: W, H: H,
                                                                mainIdx: mainIdx,
                                                                cropRectNorm: self.centerCropMetaRect) {
                    let N = self.lastApertureN
                    let userS = max(25.0, min(204800.0, self.userISO))
                    let evTarget = self.lastEV100 - deltaEV
                    let tUser = self.solveShutter(ev100: evTarget, n: N, s: userS)
                    let areaPct = self.percentArea(of: fusedIndex, cls: mainIdx) * 100.0
                    let biasPct = Int((min(1.0, w)) * 100.0)
                    let text = "Main-biased: \(self.fmtAperture(N)) · \(self.fmtShutter(tUser)) @ ISO \(Int(userS))  (\(Int(areaPct))% area · bias \(biasPct)%)"
                    DispatchQueue.main.async { [weak self] in
                        guard let self else { return }
                        self.meterSubject = text
                        self.pushToWatch()
                    }
                }
            }

            DispatchQueue.main.async {
                self.overlayImage = finalImg
                self.classPercentages = finalPct
                self.lastUpdateText = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
                self.mainSubjectText = "Main: \(mainName)"
                self.mainClassIndex = mainIdx
                if counts.reduce(0,+) == 0 { self.errorMessage = "No results from Vision." }
                self.frameGate.signal()
            }
        }
    }

    private func logAndSetError(_ msg: String) {
        log(msg)
        DispatchQueue.main.async { [weak self] in self?.errorMessage = msg; self?.frameGate.signal() }
    }

    // === SUBJECTNESS === utilities

    private func semanticPrior(for label: String) -> Double {
        let s = label.lowercased()
        if s.contains("face") { return 1.0 }
        if s.contains("person") || s.contains("people") || s.contains("human") { return 0.9 }
        if s.contains("cat") || s.contains("dog") || s.contains("animal") || s.contains("bird") { return 0.8 }
        if s.contains("car") || s.contains("vehicle") || s.contains("bike") { return 0.6 }
        if s.contains("sky") || s.contains("road") || s.contains("wall") || s.contains("floor") || s.contains("background") { return 0.1 }
        return 0.5
    }

    private func pickMainSubject(W: Int, H: Int,
                                 counts: [Int],
                                 saliencySum: [Double],
                                 centroid: [(Double, Double)]) -> (mainIdx: Int, mainName: String) {
        let totalPx = max(1, counts.reduce(0,+))
        let cx = Double(W - 1) * 0.5, cy = Double(H - 1) * 0.5
        let diag = sqrt(Double(W*W + H*H))
        let sigma = 0.25 * diag

        var bestIdx = -1
        var bestScore = -Double.greatestFiniteMagnitude

        for i in 0..<counts.count {
            let cnt = counts[i]
            if cnt == 0 { continue }

            let name = i < labels.count ? labels[i] : "class_\(i)"
            let prior = semanticPrior(for: name)
            let salAvg = saliencySum[i] / Double(max(1, cnt))
            let size = min(Double(cnt)/Double(totalPx), 0.3)
            let (mx,my) = centroid[i]
            let d = sqrt((mx - cx)*(mx - cx) + (my - cy)*(my - cy))
            let geo = exp(-d / sigma)

            let score = 0.50*salAvg + 0.25*geo + 0.15*prior + 0.10*size
            if score > bestScore { bestScore = score; bestIdx = i }
        }
        let mainName = bestIdx >= 0 ? cleanLabel(bestIdx < labels.count ? labels[bestIdx] : "class_\(bestIdx)") : "—"
        return (bestIdx, mainName)
    }

    private func makeSaliencySampler(_ pb: CVPixelBuffer?, targetW: Int, targetH: Int) -> ((Int, Int) -> Double)? {
        guard let pb else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let sw = CVPixelBufferGetWidth(pb)
        let sh = CVPixelBufferGetHeight(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(pb)

        let isGray8 = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent8
        let sx = Double(sw) / Double(targetW)
        let sy = Double(sh) / Double(targetH)

        return { (x: Int, y: Int) -> Double in
            let sxI = min(sw-1, max(0, Int((Double(x)+0.5)*sx)))
            let syI = min(sh-1, max(0, Int((Double(y)+0.5)*sy)))
            let row = base.advanced(by: syI * bpr)
            if isGray8 {
                let p = row.bindMemory(to: UInt8.self, capacity: bpr)
                return Double(p[sxI]) / 255.0
            } else {
                let p = row.bindMemory(to: UInt8.self, capacity: bpr)
                let bgra = p + sxI*4
                let b = Double(bgra[0]), g = Double(bgra[1]), r = Double(bgra[2])
                return linearLumaFromBGRA(b, g, r)
            }
        }
    }

    // MARK: - Metering helpers (EV math + formatting)

    private func fmtShutter(_ t: Double) -> String {
        if t <= 0 { return "—" }
        if t < 1.0 {
            let denom = max(1, Int((1.0 / t).rounded()))
            return "1/\(denom)s"
        } else {
            let val = (t * 10).rounded() / 10.0
            return "\(val)s"
        }
    }
    private func fmtAperture(_ n: Double) -> String {
        String(format: "f/%.1f", n)
    }
    private func ev100(n: Double, t: Double, s: Double) -> Double {
        guard n > 0, t > 0, s > 0 else { return .nan }
        return log2((n*n)/t) - log2(s/100.0)
    }
    private func solveShutter(ev100: Double, n: Double, s: Double) -> Double {
        let denom = pow(2.0, ev100) * (s/100.0)
        return max(1e-6, (n*n) / denom)
    }
    private func solveAperture(ev100: Double, t: Double, s: Double) -> Double {
        let val = t * pow(2.0, ev100) * (s/100.0)
        return sqrt(max(1e-8, val))
    }

    // === Postprocess dispatch ===

    private func postprocessFeatureValue(_ obs: VNCoreMLFeatureValueObservation,
                                         saliencyPB: CVPixelBuffer?)
    -> (overlay: CGImage, percentages: [(String, Double, Int)], mainName: String, mainIdx: Int, indexMap: [Int], W: Int, H: Int)? {
        guard let arr = obs.featureValue.multiArrayValue else {
            log("FeatureValue not MultiArray")
            return nil
        }
        let shape = arr.shape.map { Int(truncating: $0) }
        switch shape.count {
        case 3: return postprocessLogitsCHW(arr, saliencyPB: saliencyPB)
        case 2: return postprocessLabelMapHW(arr, saliencyPB: saliencyPB)
        default:
            log("Unsupported MultiArray rank \(shape.count) (expect 2 or 3)")
            return nil
        }
    }

    // logits [C,H,W]
    private func postprocessLogitsCHW(_ arr: MLMultiArray,
                                      saliencyPB: CVPixelBuffer?)
    -> (overlay: CGImage, percentages: [(String, Double, Int)], mainName: String, mainIdx: Int, indexMap: [Int], W: Int, H: Int)? {
        let shape = arr.shape.map { Int(truncating: $0) }
        guard shape.count == 3 else { return nil }
        let C = shape[0], H = shape[1], W = shape[2]
        let pxCount = W * H

        if labels.count < max(1000, C) {
            labels = padLabels(labels, toAtLeast: max(1000, C))
        }

        var counts = Array(repeating: 0, count: C)
        var overlay = [UInt32](repeating: 0, count: pxCount)
        var indexMap = [Int](repeating: 0, count: pxCount)

        var saliencySum = Array(repeating: 0.0, count: C)
        var sumX = Array(repeating: 0.0, count: C)
        var sumY = Array(repeating: 0.0, count: C)
        let salSample = makeSaliencySampler(saliencyPB, targetW: W, targetH: H)

        let sC = arr.strides[0].intValue
        let sH = arr.strides[1].intValue
        let sW = arr.strides[2].intValue

        switch arr.dataType {
        case .float32:
            let p = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let base = y*sH + x*sW
                    var best = 0
                    var bestV: Float32 = -Float.greatestFiniteMagnitude
                    for c in 0..<C {
                        let v = p[base + c*sC]
                        if v > bestV { bestV = v; best = c }
                    }
                    counts[best] += 1
                    indexMap[y*W + x] = best
                    if let s = salSample { saliencySum[best] += s(x,y) }
                    sumX[best] += Double(x); sumY[best] += Double(y)
                }
            }
        case .double:
            let p = UnsafeMutablePointer<Double>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let base = y*sH + x*sW
                    var best = 0
                    var bestV: Double = -Double.greatestFiniteMagnitude
                    for c in 0..<C {
                        let v = p[base + c*sC]
                        if v > bestV { bestV = v; best = c }
                    }
                    counts[best] += 1
                    indexMap[y*W + x] = best
                    if let s = salSample { saliencySum[best] += s(x,y) }
                    sumX[best] += Double(x); sumY[best] += Double(y)
                }
            }
        case .float16:
            let p = UnsafeMutablePointer<UInt16>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let base = y*sH + x*sW
                    var best = 0
                    var bestV = -Float.greatestFiniteMagnitude
                    for c in 0..<C {
                        let bits = p[base + c*sC]
                        let v = Float(Float16(bitPattern: bits))
                        if v > bestV { bestV = v; best = c }
                    }
                    counts[best] += 1
                    indexMap[y*W + x] = best
                    if let s = salSample { saliencySum[best] += s(x,y) }
                    sumX[best] += Double(x); sumY[best] += Double(y)
                }
            }
        default:
            log("Unsupported logits dataType \(arr.dataType)")
            return nil
        }

        var centroid = Array(repeating: (0.0, 0.0), count: C)
        for i in 0..<C {
            let cnt = max(1, counts[i])
            centroid[i] = (sumX[i]/Double(cnt), sumY[i]/Double(cnt))
        }

        let (mainIdx, mainName) = pickMainSubject(W: W, H: H, counts: counts, saliencySum: saliencySum, centroid: centroid)

        for i in 0..<indexMap.count {
            let cls = indexMap[i]
            overlay[i] = pixelBGRA(for: cls, boostMain: cls == mainIdx)
        }

        let (img, pct) = finalizeOverlay(W: W, H: H, counts: counts, overlay: &overlay)
        return (img, pct, mainName, mainIdx, indexMap, W, H)
    }

    // label map [H,W]
    private func postprocessLabelMapHW(_ arr: MLMultiArray,
                                       saliencyPB: CVPixelBuffer?)
    -> (overlay: CGImage, percentages: [(String, Double, Int)], mainName: String, mainIdx: Int, indexMap: [Int], W: Int, H: Int)? {
        let shape = arr.shape.map { Int(truncating: $0) }
        guard shape.count == 2 else { return nil }
        let H = shape[0], W = shape[1]

        var overlay = [UInt32](repeating: 0, count: W * H)
        var indexMap = [Int](repeating: 0, count: W * H)
        var counts: [Int] = []
        var maxClsSeen = -1

        var saliencySum: [Double] = []
        var sumX: [Double] = []
        var sumY: [Double] = []
        let salSample = makeSaliencySampler(saliencyPB, targetW: W, targetH: H)

        func ensureCapacity(_ cls: Int) {
            if cls >= counts.count {
                let old = counts.count
                counts.append(contentsOf: Array(repeating: 0, count: cls - old + 1))
                saliencySum.append(contentsOf: Array(repeating: 0.0, count: cls - old + 1))
                sumX.append(contentsOf: Array(repeating: 0.0, count: cls - old + 1))
                sumY.append(contentsOf: Array(repeating: 0.0, count: cls - old + 1))
            }
        }

        let sH = arr.strides[0].intValue
        let sW = arr.strides[1].intValue

        switch arr.dataType {
        case .int32:
            let p = UnsafeMutablePointer<Int32>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let cls = Int(p[y*sH + x*sW])
                    if cls >= 0 {
                        ensureCapacity(cls)
                        counts[cls] += 1
                        indexMap[y*W + x] = cls
                        if let s = salSample { saliencySum[cls] += s(x,y) }
                        sumX[cls] += Double(x); sumY[cls] += Double(y)
                        if cls > maxClsSeen { maxClsSeen = cls }
                    }
                }
            }
        case .float32:
            let p = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let cls = max(0, Int(p[y*sH + x*sW].rounded()))
                    ensureCapacity(cls)
                    counts[cls] += 1
                    indexMap[y*W + x] = cls
                    if let s = salSample { saliencySum[cls] += s(x,y) }
                    sumX[cls] += Double(x); sumY[cls] += Double(y)
                    if cls > maxClsSeen { maxClsSeen = cls }
                }
            }
        case .float16:
            let p = UnsafeMutablePointer<UInt16>(OpaquePointer(arr.dataPointer))
            for y in 0..<H {
                for x in 0..<W {
                    let bits = p[y*sH + x*sW]
                    let cls = max(0, Int(Float(Float16(bitPattern: bits)).rounded()))
                    ensureCapacity(cls)
                    counts[cls] += 1
                    indexMap[y*W + x] = cls
                    if let s = salSample { saliencySum[cls] += s(x,y) }
                    sumX[cls] += Double(x); sumY[cls] += Double(y)
                    if cls > maxClsSeen { maxClsSeen = cls }
                }
            }
        default:
            log("Unsupported label-map dataType \(arr.dataType)")
            return nil
        }

        if labels.count <= maxClsSeen || labels.count < 1000 {
            labels = padLabels(labels, toAtLeast: max(1000, maxClsSeen + 1))
        }

        var centroid = Array(repeating: (0.0, 0.0), count: counts.count)
        for i in 0..<counts.count {
            let cnt = max(1, counts[i])
            centroid[i] = (sumX[i]/Double(cnt), sumY[i]/Double(cnt))
        }

        let (mainIdx, mainName) = pickMainSubject(W: W, H: H, counts: counts, saliencySum: saliencySum, centroid: centroid)

        for y in 0..<H {
            for x in 0..<W {
                let cls = indexMap[y*W + x]
                overlay[y*W + x] = pixelBGRA(for: cls, boostMain: cls == mainIdx)
            }
        }

        let (img, pct) = finalizeOverlay(W: W, H: H, counts: counts, overlay: &overlay)
        log("Observed classes 0...\(maxClsSeen >= 0 ? maxClsSeen : 0); counts size \(counts.count); labels \(labels.count)")
        return (img, pct, mainName, mainIdx, indexMap, W, H)
    }

    private func postprocessClassIndexMap(_ pb: CVPixelBuffer,
                                          saliencyPB: CVPixelBuffer?)
    -> (overlay: CGImage, percentages: [(String, Double, Int)], mainName: String, mainIdx: Int, indexMap: [Int], W: Int, H: Int)? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let W = CVPixelBufferGetWidth(pb)
        let H = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }

        var counts: [Int] = []
        var indexMap = [Int](repeating: 0, count: W * H)
        var maxClsSeen = -1

        var saliencySum: [Double] = []
        var sumX: [Double] = []
        var sumY: [Double] = []
        let salSample = makeSaliencySampler(saliencyPB, targetW: W, targetH: H)

        func ensureCapacity(_ cls: Int) {
            if cls >= counts.count {
                let add = cls - counts.count + 1
                counts.append(contentsOf: repeatElement(0, count: add))
                saliencySum.append(contentsOf: repeatElement(0.0, count: add))
                sumX.append(contentsOf: repeatElement(0.0, count: add))
                sumY.append(contentsOf: repeatElement(0.0, count: add))
            }
        }

        for y in 0..<H {
            let row = base.advanced(by: y * bpr)
            let ptr = row.bindMemory(to: UInt8.self, capacity: bpr)
            for x in 0..<W {
                let cls = Int(ptr[x])
                ensureCapacity(cls)
                counts[cls] &+= 1
                indexMap[y*W + x] = cls
                if let s = salSample { saliencySum[cls] += s(x,y) }
                sumX[cls] += Double(x); sumY[cls] += Double(y)
                if cls > maxClsSeen { maxClsSeen = cls }
            }
        }

        if labels.count <= maxClsSeen || labels.count < 1000 {
            labels = padLabels(labels, toAtLeast: max(1000, maxClsSeen + 1))
        }

        var centroid = Array(repeating: (0.0, 0.0), count: counts.count)
        for i in 0..<counts.count {
            let cnt = max(1, counts[i])
            centroid[i] = (sumX[i]/Double(cnt), sumY[i]/Double(cnt))
        }

        let (mainIdx, mainName) = pickMainSubject(W: W, H: H,
                                                  counts: counts,
                                                  saliencySum: saliencySum,
                                                  centroid: centroid)

        var overlay = [UInt32](repeating: 0, count: W * H)
        for i in 0..<indexMap.count {
            let cls = indexMap[i]
            overlay[i] = pixelBGRA(for: cls, boostMain: cls == mainIdx)
        }

        let (img, pct) = finalizeOverlay(W: W, H: H, counts: counts, overlay: &overlay)
        log("PixelBuffer result: \(W)x\(H); observed classes up to \(maxClsSeen)")
        return (img, pct, mainName, mainIdx, indexMap, W, H)
    }

    // MARK: - 主体偏置 ΔEV（18% 灰 + 面积权重）

    private func subjectBiasedDeltaEV(from pb: CVPixelBuffer,
                                      mask: [Int], W: Int, H: Int,
                                      mainIdx: Int,
                                      cropRectNorm: CGRect) -> (deltaEV: Double, biasW: Double)? {
        guard mainIdx >= 0, mask.count == W*H else { return nil }

        var histMain = [Int](repeating: 0, count: 256)
        var histOther = [Int](repeating: 0, count: 256)
        var mainCount = 0
        var otherCount = 0

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let srcW = CVPixelBufferGetWidth(pb)
        let srcH = CVPixelBufferGetHeight(pb)
        let bpr  = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let p = base.bindMemory(to: UInt8.self, capacity: bpr * srcH)

        // map center-crop
        let cropX = cropRectNorm.origin.x * Double(srcW)
        let cropY = cropRectNorm.origin.y * Double(srcH)
        let cropW = cropRectNorm.width  * Double(srcW)
        let cropH = cropRectNorm.height * Double(srcH)

        let stepX = max(1, W / 160)
        let stepY = max(1, H / 160)

        for y in stride(from: 0, to: H, by: stepY) {
            let sy = min(srcH - 1, max(0, Int(cropY + (Double(y) + 0.5) / Double(H) * cropH)))
            let row = p + sy * bpr
            for x in stride(from: 0, to: W, by: stepX) {
                let sx = min(srcW - 1, max(0, Int(cropX + (Double(x) + 0.5) / Double(W) * cropW)))
                let bgra = row + sx * 4
                let b = Double(bgra[0]), g = Double(bgra[1]), r = Double(bgra[2])
                let yLin = linearLumaFromBGRA(b, g, r)
                let bin = Int(min(255.0, max(0.0, yLin * 255.0)).rounded())

                if mask[y*W + x] == mainIdx {
                    histMain[bin] &+= 1
                    mainCount &+= 1
                } else {
                    histOther[bin] &+= 1
                    otherCount &+= 1
                }
            }
        }

        guard mainCount > 0 else { return (0, 0) }

        func percentile(_ hist: [Int], _ total: Int, _ p: Double) -> Double {
            if total == 0 { return .nan }
            let rank = max(0, min(total - 1, Int(Double(total - 1) * p + 0.5)))
            var acc = 0
            for i in 0..<256 {
                acc += hist[i]
                if acc > rank { return Double(i) / 255.0 }
            }
            return 1.0
        }

        let mainMed = percentile(histMain, mainCount, 0.5)
        let deltaSubjectEV = log2(max(1e-5, targetGray) / max(1e-5, mainMed))

        let area = percentArea(of: mask, cls: mainIdx)
        let w = min(biasMax, pow(area, biasExponent))
        var deltaEV = w * deltaSubjectEV

        if useClipGuards, otherCount > 0 {
            let otherP5  = percentile(histOther, otherCount, 0.05)
            let otherP95 = percentile(histOther, otherCount, 0.95)
            var hiBound = Double.infinity
            var loBound = -Double.infinity
            if otherP95 > 0 { hiBound = log2(clipHigh / max(1e-5, otherP95)) }
            if otherP5  > 0 { loBound = log2(clipLow  / max(1e-5, otherP5 )) }
            deltaEV = min(max(deltaEV, loBound), hiBound)
        }

        return (deltaEV, w)
    }

    private func percentArea(of mask: [Int], cls: Int) -> Double {
        if cls < 0 { return 0 }
        let total = mask.count
        if total == 0 { return 0 }
        let c = mask.reduce(0) { $1 == cls ? $0 + 1 : $0 }
        return Double(c) / Double(total)
    }

    // MARK: - 分区测光（18% 灰）

    private func zonalDeltaEV(from pb: CVPixelBuffer,
                              cropRectNorm: CGRect) -> Double? {

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }

        let sw = CVPixelBufferGetWidth(pb)
        let sh = CVPixelBufferGetHeight(pb)
        let bpr = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let px = base.bindMemory(to: UInt8.self, capacity: bpr * sh)

        let cx = cropRectNorm.origin.x * Double(sw)
        let cy = cropRectNorm.origin.y * Double(sh)
        let cw = cropRectNorm.width  * Double(sw)
        let ch = cropRectNorm.height * Double(sh)

        let stepX = max(1, Int(cw / Double(zoneCols) / 24.0))
        let stepY = max(1, Int(ch / Double(zoneRows) / 24.0))

        var deltas: [(Double, Double)] = []

        for gy in 0..<zoneRows {
            for gx in 0..<zoneCols {
                let x0 = Int(cx + (Double(gx)   / Double(zoneCols)) * cw)
                let x1 = Int(cx + (Double(gx+1) / Double(zoneCols)) * cw)
                let y0 = Int(cy + (Double(gy)   / Double(zoneRows)) * ch)
                let y1 = Int(cy + (Double(gy+1) / Double(zoneRows)) * ch)

                var hist = [Int](repeating: 0, count: 256)
                var count = 0

                var yy = y0
                while yy < y1 {
                    let row = px + yy * bpr
                    var xx = x0
                    while xx < x1 {
                        let bgra = row + xx * 4
                        let b = Double(bgra[0]), g = Double(bgra[1]), r = Double(bgra[2])
                        let yLin = linearLumaFromBGRA(b, g, r)
                        let bin = Int(min(255.0, max(0.0, yLin * 255.0)).rounded())
                        hist[bin] &+= 1
                        count &+= 1
                        xx += stepX
                    }
                    yy += stepY
                }

                guard count > 0 else { continue }

                let med = percFromHist(hist, total: count, p: 0.5)
                let p5  = percFromHist(hist, total: count, p: 0.05)
                let p95 = percFromHist(hist, total: count, p: 0.95)

                var dEV = log2(max(1e-5, zoneTargetGray) / max(1e-5, med))
                var hi = Double.infinity, lo = -Double.infinity
                if p95 > 0 { hi = log2(zoneClipHigh / max(1e-5, p95)) }
                if p5  > 0 { lo = log2(zoneClipLow  / max(1e-5, p5 )) }
                dEV = min(max(dEV, lo), hi)

                let w = Double(count)
                deltas.append((dEV, w))
            }
        }

        guard !deltas.isEmpty else { return nil }
        return weightedMedian(deltas)
    }

    @inline(__always)
    private func percFromHist(_ hist: [Int], total: Int, p: Double) -> Double {
        guard total > 0 else { return .nan }
        let rank = max(0, min(total - 1, Int(Double(total - 1) * p + 0.5)))
        var acc = 0
        for i in 0..<256 {
            acc += hist[i]
            if acc > rank { return Double(i) / 255.0 }
        }
        return 1.0
    }

    private func weightedMedian(_ pairs: [(Double, Double)]) -> Double {
        let totalW = pairs.reduce(0.0) { $0 + $1.1 }
        if totalW <= 0 { return 0 }
        let sorted = pairs.sorted { $0.0 < $1.0 }
        var acc = 0.0
        let half = totalW * 0.5
        for (v, w) in sorted {
            acc += w
            if acc >= half { return v }
        }
        return sorted.last?.0 ?? 0
    }

    // MARK: - Finalization & helpers

    private func finalizeOverlay(W: Int, H: Int,
                                 counts: [Int],
                                 overlay: inout [UInt32]) -> (CGImage, [(String, Double, Int)]) {
        let totalPx = max(1, overlay.count)
        var percentages: [(String, Double, Int)] = []
        percentages.reserveCapacity(counts.count)

        for i in 0..<counts.count {
            let cnt = counts[i]
            if cnt == 0 { continue }
            let rawName = i < labels.count ? labels[i] : "class_\(i)"
            let name = cleanLabel(rawName)
            let pct = (Double(cnt) / Double(totalPx)) * 100.0
            percentages.append((name, pct, i))
        }

        percentages.sort { $0.1 > $1.1 }
        let mask = makeCGImageBGRA(width: W, height: H, data: &overlay)!
        return (mask, percentages)
    }

    private func makeCGImageBGRA(width: Int, height: Int, data: inout [UInt32]) -> CGImage? {
        let bytesPerRow = width * 4
        return data.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return nil }
            let provider = CGDataProvider(dataInfo: nil, data: base, size: raw.count) { _,_,_ in }!
            return CGImage(width: width, height: height,
                           bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
                           space: CGColorSpaceCreateDeviceRGB(),
                           bitmapInfo: CGBitmapInfo.byteOrder32Little.union(
                               CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
                           ),
                           provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        }
    }

    @inline(__always)
    private func majority(_ a: Int, _ b: Int) -> Int { a == b ? a : a }

    private func flipIndexMapHoriz(_ map: [Int], W: Int, H: Int) -> [Int] {
        var out = map
        for y in 0..<H {
            let row = y * W
            for x in 0..<(W / 2) {
                let l = row + x
                let r = row + (W - 1 - x)
                out.swapAt(l, r)
            }
        }
        return out
    }

    private func mode3x3(_ idx: inout [Int], W: Int, H: Int) {
        var out = idx
        for y in 1..<(H-1) {
            for x in 1..<(W-1) {
                var hist = [Int: Int]()
                for dy in -1...1 {
                    for dx in -1...1 {
                        let c = idx[(y+dy)*W + (x+dx)]
                        hist[c, default: 0] += 1
                    }
                }
                if let best = hist.max(by: { $0.value < $1.value })?.key {
                    out[y*W + x] = best
                }
            }
        }
        idx = out
    }

    private func overlayFromIndexMap(indexMap: [Int], W: Int, H: Int, mainIdx: Int)
    -> (CGImage, [(String, Double, Int)], [Int]) {
        var counts: [Int] = []
        counts.reserveCapacity(128)
        var overlay = [UInt32](repeating: 0, count: W*H)
        var maxCls = -1
        for i in 0..<indexMap.count {
            let cls = indexMap[i]
            if cls >= counts.count {
                counts.append(contentsOf: repeatElement(0, count: cls - counts.count + 1))
            }
            counts[cls] += 1
            if cls > maxCls { maxCls = cls }
            overlay[i] = pixelBGRA(for: cls, boostMain: cls == mainIdx)
        }
        if labels.count <= maxCls || labels.count < 1000 {
            labels = padLabels(labels, toAtLeast: max(1000, maxCls + 1))
        }
        let (img, pct) = finalizeOverlay(W: W, H: H, counts: counts, overlay: &overlay)
        return (img, pct, counts)
    }
}

extension ViewModel: @unchecked Sendable {}

extension ViewModel: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if frameGate.wait(timeout: .now()) != .success { return }

        frameCounter &+= 1
        if frameSkip > 0, frameCounter % (frameSkip + 1) != 0 {
            frameGate.signal(); return
        }

        // center-crop rect in video coords
        let srcW = CGFloat(CVPixelBufferGetWidth(pb))
        let srcH = CGFloat(CVPixelBufferGetHeight(pb))
        var metaRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        if srcW > srcH {
            let wNorm = srcH / srcW
            metaRect = CGRect(x: (1 - wNorm) * 0.5, y: 0, width: wNorm, height: 1)
        } else if srcH > srcW {
            let hNorm = srcW / srcH
            metaRect = CGRect(x: 0, y: (1 - hNorm) * 0.5, width: 1, height: hNorm)
        }
        DispatchQueue.main.async { [weak self] in
            self?.centerCropMetaRect = metaRect
        }

        // orientation
        let angleDeg: Double
        if #available(iOS 17.0, macOS 14.0, *) { angleDeg = connection.videoRotationAngle }
        else {
            #if os(iOS)
            switch connection.videoOrientation {
            case .portrait: angleDeg = 90
            case .portraitUpsideDown: angleDeg = 270
            case .landscapeRight: angleDeg = 0
            case .landscapeLeft: angleDeg = 180
            @unknown default: angleDeg = 0
            }
            #else
            angleDeg = 0
            #endif
        }
        let mirrored = connection.isVideoMirrored
        let exif = exifFrom(angleDegrees: angleDeg, mirrored: mirrored)

        // === Meter A: 设备曝光 → EV → 用户 ISO 等效
        if let dev = self.activeDevice {
            let N = Double(dev.lensAperture)
            let t = Double(CMTimeGetSeconds(dev.exposureDuration))
            let S = Double(dev.iso)
            if N > 0 && t > 0 && S > 0 {
                let EV100 = ev100(n: N, t: t, s: S)
                self.lastEV100 = EV100
                self.lastApertureN = N

                let userS = max(25.0, min(204800.0, self.userISO))
                let tUser = solveShutter(ev100: EV100, n: N, s: userS)
                let main = "\(fmtAperture(N)) · \(fmtShutter(tUser)) @ ISO \(Int(userS))"

                let tTarget = self.targetShutterForApertureSuggestions
                let nSolve = solveAperture(ev100: EV100, t: tTarget, s: userS)
                let altsTuples: [(n: Double, t: Double)] = self.fullStops.map { stop in
                    let tAlt = solveShutter(ev100: EV100, n: stop, s: userS)
                    return (stop, tAlt)
                }
                let sorted = altsTuples.sorted { abs($0.n - nSolve) < abs($1.n - nSolve) }
                let alts = sorted.prefix(5).map { "\(fmtAperture($0.n)) · \(fmtShutter($0.t))" }

                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.meterMain = main
                    self.meterAlternates = Array(alts)
                    self.pushToWatch()
                }
            }
        }

        // === Meter C: Zonal（18% 灰；置顶）
        if !self.lastEV100.isNaN {
            if let dEVz = self.zonalDeltaEV(from: pb, cropRectNorm: self.centerCropMetaRect) {
                let N = self.lastApertureN
                let userS = max(25.0, min(204800.0, self.userISO))
                let evTarget = self.lastEV100 - dEVz
                let tUser = self.solveShutter(ev100: evTarget, n: N, s: userS)
                let txt = "Zonal: \(self.fmtAperture(N)) · \(self.fmtShutter(tUser)) @ ISO \(Int(userS))"
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.meterZonal = txt
                    self.pushToWatch()
                }
            }
        }

        // Run segmentation; subject-biased meter updated after inference
        runInference(on: pb, exif: exif)
    }
}

// MARK: - SwiftUI root + HUD

struct SegmentedCameraRootView: View {
    @StateObject private var vm = ViewModel()

    private func startWatchSync() {
        PhoneWatchConnectivity.shared.start { newISO in
            print("[VM] userISO <- \(newISO)")
            vm.userISO = max(25.0, min(204800.0, newISO))
        }
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            #if os(iOS)
            CameraView_iOS(session: vm.session,
                           overlayImage: vm.overlayImage,
                           centerCropMetaRect: vm.centerCropMetaRect)
                .ignoresSafeArea()
            #else
            CameraView_macOS(session: vm.session, overlayImage: vm.overlayImage)
            #endif

            HUDView(vm: vm,
                    percentages: vm.classPercentages,
                    error: vm.errorMessage,
                    last: vm.lastUpdateText,
                    mainText: vm.mainSubjectText,
                    mainClassIndex: vm.mainClassIndex)
                .padding()
        }
        .task { await vm.loadModel() }
        .onAppear {
            vm.requestAuthorizationAndStart()
            startWatchSync()     // 启动 iPhone ↔︎ Watch 通讯
        }
    }
}

struct HUDView: View {
    @ObservedObject var vm: ViewModel

    let percentages: [(label: String, percent: Double, cls: Int)]
    let error: String?
    let last: String
    let mainText: String
    let mainClassIndex: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // ISO
            HStack(spacing: 8) {
                Text("ISO").foregroundColor(.white)
                TextField("100", value: $vm.userISO, format: .number)
                    .frame(width: 72)
                    .textFieldStyle(.roundedBorder)
                #if os(iOS)
                    .keyboardType(.numberPad)
                #endif
            }

            // C) Zonal（置顶）
            Text(vm.meterZonal)
                .foregroundColor(.white)
                .font(.subheadline)
                .bold()

            // A) Scene-equivalent
            Text(vm.meterMain)
                .foregroundColor(.white).font(.subheadline)

            if !vm.meterAlternates.isEmpty {
                Text(vm.meterAlternates.joined(separator: "  •  "))
                    .foregroundColor(.white.opacity(0.9))
                    .font(.caption2)
                    .lineLimit(2)
            }

            // B) Main-biased
            Text(vm.meterSubject)
                .foregroundColor(.white)
                .font(.subheadline)

            Divider().background(Color.white.opacity(0.3))

            // Segmentation HUD
            if let error {
                Text(error).foregroundColor(.red)
            } else if percentages.isEmpty {
                Text("Running segmentation…").foregroundColor(.white)
            } else {
                Text(mainText).bold().foregroundColor(.yellow)
                Text("Top classes:").bold().foregroundColor(.white)
                let top = Array(percentages.prefix(6))
                ForEach(top.indices, id: \.self) { i in
                    let item = top[i]
                    let isMain = (item.cls == mainClassIndex)
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(colorForClassSwiftUI(item.cls, boost: isMain))
                            .frame(width: 12, height: 12)
                        Text(item.label).lineLimit(1)
                        Spacer(minLength: 8)
                        Text("\(item.percent, specifier: "%.1f")%").monospacedDigit()
                    }
                    .foregroundColor(.white)
                }
                if !last.isEmpty {
                    Text("Last update: \(last)").font(.caption2).foregroundColor(.white.opacity(0.85))
                }
            }
        }
        .padding(8)
        .background(Color.black.opacity(0.5))
        .cornerRadius(8)
    }
}

// MARK: - Platform wrappers

#if os(iOS)
import UIKit

final class PreviewView_iOS: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    let overlayLayer = CALayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .black
        videoLayer.videoGravity = .resizeAspectFill

        overlayLayer.contentsGravity = .resizeAspectFill
        overlayLayer.isOpaque = false
        overlayLayer.masksToBounds = true
        overlayLayer.opacity = 1.0
        overlayLayer.compositingFilter = "sourceOver"
        overlayLayer.contentsScale = UIScreen.main.scale

        layer.addSublayer(overlayLayer)
    }

    required init?(coder: NSCoder) { super.init(coder: coder) }

    override func layoutSubviews() {
        super.layoutSubviews()
    }
}

struct CameraView_iOS: UIViewRepresentable {
    let session: AVCaptureSession
    let overlayImage: CGImage?
    let centerCropMetaRect: CGRect
    var rotateOverlayCW90: Bool = true

    func makeUIView(context: Context) -> PreviewView_iOS {
        let v = PreviewView_iOS()
        v.videoLayer.session = session
        return v
    }

    func updateUIView(_ uiView: PreviewView_iOS, context: Context) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)

        uiView.overlayLayer.contents = overlayImage
        uiView.overlayLayer.contentsScale = UIScreen.main.scale
        uiView.overlayLayer.contentsGravity = .resizeAspectFill

        let layerRect = uiView.videoLayer.layerRectConverted(fromMetadataOutputRect: centerCropMetaRect)
        uiView.overlayLayer.frame = layerRect

        var t = CGAffineTransform.identity
        if rotateOverlayCW90 { t = t.rotated(by: .pi / 2) }
        if let conn = uiView.videoLayer.connection, conn.isVideoMirrored {
            t = t.scaledBy(x: -1, y: 1)
        }
        uiView.overlayLayer.setAffineTransform(t)

        CATransaction.commit()
    }
}
#endif

#if os(macOS)
import AppKit

final class PreviewView_macOS: NSView {
    let videoLayer = AVCaptureVideoPreviewLayer()
    let overlayLayer = CALayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.addSublayer(videoLayer)
        layer?.addSublayer(overlayLayer)

        videoLayer.videoGravity = .resizeAspectFill

        overlayLayer.contentsGravity = .resizeAspectFill
        overlayLayer.isOpaque = false
        overlayLayer.masksToBounds = true
        overlayLayer.opacity = 1.0
        overlayLayer.compositingFilter = "sourceOver"
        overlayLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
    }

    required init?(coder: NSCoder) { super.init(coder: coder); wantsLayer = true }

    override func layout() {
        super.layout()
        videoLayer.frame = bounds
        overlayLayer.frame = bounds
    }
}

struct CameraView_macOS: NSViewRepresentable {
    let session: AVCaptureSession
    let overlayImage: CGImage?

    func makeNSView(context: Context) -> PreviewView_macOS {
        let v = PreviewView_macOS()
        v.videoLayer.session = session
        return v
    }

    func updateNSView(_ nsView: PreviewView_macOS, context: Context) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)

        nsView.overlayLayer.contents = overlayImage
        nsView.overlayLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
        nsView.overlayLayer.frame = nsView.bounds
        nsView.overlayLayer.contentsRect = nsView.videoLayer.contentsRect
        nsView.overlayLayer.setAffineTransform(.identity)

        CATransaction.commit()
    }
}
#endif
