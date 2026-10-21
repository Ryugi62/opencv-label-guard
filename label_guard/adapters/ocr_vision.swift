#!/usr/bin/env swift
// 이미지 OCR (macOS Vision, 온디바이스, 외부 전송 0). ko+en.
// usage: ocr_vision.swift <img1> [img2 ...]  → 각 이미지마다 JSON 한 줄
//   {"file":..., "lines":[{"text":..., "x":0-1, "y":0-1(위=0), "w":..., "h":...}], "text_area":0-1}
// text_area = 인식된 텍스트 박스 면적 합/이미지 면적 (작으면 그림 위주 슬라이드)
import Foundation
import Vision
import AppKit

func ocr(_ path: String) -> [String: Any] {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        return ["file": path, "error": "load failed"]
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.recognitionLanguages = ["ko-KR", "en-US"]
    req.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do { try handler.perform([req]) } catch { return ["file": path, "error": "\(error)"] }
    var lines: [[String: Any]] = []
    var area = 0.0
    for obs in req.results ?? [] {
        guard let top = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox  // 원점 좌하단
        lines.append(["text": top.string, "x": b.minX, "y": 1 - b.maxY, "w": b.width, "h": b.height, "conf": top.confidence])
        area += Double(b.width * b.height)
    }
    // 위→아래, 왼→오 정렬
    lines.sort { (a, b) in
        let ay = a["y"] as! CGFloat, by = b["y"] as! CGFloat
        if abs(ay - by) > 0.015 { return ay < by }
        return (a["x"] as! CGFloat) < (b["x"] as! CGFloat)
    }
    return ["file": path, "lines": lines, "text_area": area]
}

let args = Array(CommandLine.arguments.dropFirst())
for p in args {
    let d = ocr(p)
    if let data = try? JSONSerialization.data(withJSONObject: d, options: []),
       let s = String(data: data, encoding: .utf8) {
        print(s)
    }
}
