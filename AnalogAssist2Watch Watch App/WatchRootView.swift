//
//  WatchSession.swift
//  AnalogAssist2 (watchOS)
//

import SwiftUI
import WatchConnectivity
import Combine

// MARK: - WatchConnectivity session (watchOS)

final class WatchSession: NSObject, ObservableObject, WCSessionDelegate {
    static let shared = WatchSession()

    // Incoming data from phone
    @Published var meterZonal: String = "—"
    @Published var meterScene: String = "—"
    @Published var meterSubject: String = "—"
    @Published var iso: Double = 100

    // Session state
    @Published var activation: WCSessionActivationState = .notActivated
    @Published var isReachable: Bool = false
    @Published var isCompanionInstalled: Bool = false
    @Published var isWaitingInitialPayload: Bool = true

    private override init() {
        super.init()
        guard WCSession.isSupported() else {
            print("[WC watch] WCSession not supported on this device")
            return
        }
        let s = WCSession.default
        s.delegate = self
        print("[WC watch] start(): delegate set? \(s.delegate === self)")
        s.activate()
    }

    // MARK: WCSessionDelegate (watchOS)

    func session(_ session: WCSession,
                 activationDidCompleteWith activationState: WCSessionActivationState,
                 error: Error?) {
        DispatchQueue.main.async {
            self.activation = activationState
            self.isReachable = session.isReachable
            self.isCompanionInstalled = session.isCompanionAppInstalled
        }
        if let e = error {
            print("[WC watch] activate error: \(e.localizedDescription)")
        }
        print("[WC watch] activated state=\(activationState.rawValue) reachable=\(session.isReachable) companionInstalled=\(session.isCompanionAppInstalled)")

        // Pull whatever the phone last pushed (may be empty on first launch)
        let ctx = session.receivedApplicationContext
        if ctx.isEmpty {
            print("[WC watch] receivedApplicationContext is EMPTY on activation")
        } else {
            print("[WC watch] receivedApplicationContext on activation: \(ctx)")
        }
        applyContext(ctx) // safe even if empty
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async { self.isReachable = session.isReachable }
        print("[WC watch] reachability=\(session.isReachable)")
    }

    // watchOS: companion app install changes
    func sessionCompanionAppInstalledDidChange(_ session: WCSession) {
        DispatchQueue.main.async { self.isCompanionInstalled = session.isCompanionAppInstalled }
        print("[WC watch] companion installed=\(session.isCompanionAppInstalled)")
    }

    // Background-safe snapshot while app is running/foregrounded
    func session(_ session: WCSession,
                 didReceiveApplicationContext applicationContext: [String : Any]) {
        print("[WC watch] didReceiveApplicationContext: \(applicationContext)")
        applyContext(applicationContext)
    }

    // Foreground-only live message
    func session(_ session: WCSession, didReceiveMessage message: [String : Any]) {
        print("[WC watch] didReceiveMessage: \(message)")
        applyContext(message)
    }

    // Queued payloads (from iPhone when not reachable)
    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String : Any] = [:]) {
        print("[WC watch] didReceiveUserInfo: \(userInfo)")
        applyContext(userInfo)
    }

    // MARK: - Apply incoming data (ALWAYS on main)

    private func applyContext(_ dict: [String: Any]) {
        DispatchQueue.main.async {
            if let z = dict["meterZonal"] as? String { self.meterZonal = z }
            if let s = dict["meterScene"] as? String { self.meterScene = s }
            if let m = dict["meterSubject"] as? String { self.meterSubject = m }

            if let i = dict["iso"] as? Double {
                // Only update if meaningfully different to avoid UI churn
                if abs(i - self.iso) > 0.5 { self.iso = i }
            }

            // Once we have *any* payload, stop showing the loading UI
            if !dict.isEmpty { self.isWaitingInitialPayload = false }
        }
    }

    // MARK: - Send ISO → iPhone

    func sendISO(_ newISO: Double) {
        let vv = max(25.0, min(204800.0, newISO))
        let payload: [String: Any] = ["iso": vv]

        print("[Watch] crown/button ISO -> \(vv), reachable=\(WCSession.default.isReachable)")
        if WCSession.default.isReachable {
            WCSession.default.sendMessage(payload, replyHandler: nil) { err in
                print("[Watch] sendMessage error: \(err.localizedDescription)")
            }
        } else {
            WCSession.default.transferUserInfo(payload)
            print("[Watch] queued ISO via transferUserInfo")
        }
    }
}

// MARK: - Watch UI

struct WatchRootView: View {
    @StateObject private var wc = WatchSession.shared

    var body: some View {
        Group {
            if wc.isWaitingInitialPayload {
                VStack { ProgressView(); Text("Waiting for iPhone…").font(.footnote) }
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("ISO").font(.headline)
                            Spacer()
                            Text("\(Int(wc.iso))").font(.headline)
                        }
                        HStack {
                            Button("−") {
                                let v = max(25.0, wc.iso / sqrt(2.0))
                                wc.iso = v
                                wc.sendISO(v)
                            }
                            Button("+") {
                                let v = min(204800.0, wc.iso * sqrt(2.0))
                                wc.iso = v
                                wc.sendISO(v)
                            }
                        }
                        Divider()
                        Group {
                            Text("Zonal").font(.caption).foregroundColor(.yellow)
                            Text(wc.meterZonal).font(.callout)
                            Text("Scene").font(.caption).foregroundColor(.yellow)
                            Text(wc.meterScene).font(.callout)
                            Text("Main").font(.caption).foregroundColor(.yellow)
                            Text(wc.meterSubject).font(.callout)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 6)
                }
            }
        }
        // Crown control → send ISO deltas to iPhone
        .digitalCrownRotation(
            Binding(
                get: { wc.iso },
                set: { v in
                    let vv = max(25.0, min(204800.0, v))
                    wc.iso = vv
                    wc.sendISO(vv)
                }
            ),
            from: 25, through: 204800, by: 1,
            sensitivity: .medium, isContinuous: false, isHapticFeedbackEnabled: true
        )
    }
}
