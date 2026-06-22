#!/usr/bin/env swift

import AppKit
import Foundation

let canvasW = 308
let canvasH = 336
let gridW = 11
let gridH = 12
let patch = 28
let targetRows = [2, 5, 8, 11]
let objectCols = [4, 6, 8]
let rowSymbols = ["@", "#", "$", "&"]
let separatorLinesY = [84, 168, 252]
let objectSize: CGFloat = 22.0
let labelFontSize: CGFloat = 30.0

let shapeNames = ["circle", "square", "triangle", "diamond", "pentagon", "hexagon"]
let palette: [(String, NSColor)] = [
    ("red", NSColor(calibratedRed: 0.95, green: 0.10, blue: 0.10, alpha: 1.0)),
    ("blue", NSColor(calibratedRed: 0.15, green: 0.42, blue: 0.86, alpha: 1.0)),
    ("green", NSColor(calibratedRed: 0.14, green: 0.95, blue: 0.14, alpha: 1.0)),
    ("cyan", NSColor(calibratedRed: 0.22, green: 0.92, blue: 0.92, alpha: 1.0)),
    ("purple", NSColor(calibratedRed: 0.62, green: 0.10, blue: 0.70, alpha: 1.0)),
    ("orange", NSColor(calibratedRed: 0.93, green: 0.57, blue: 0.14, alpha: 1.0)),
    ("yellow", NSColor(calibratedRed: 0.93, green: 0.82, blue: 0.17, alpha: 1.0)),
    ("brown", NSColor(calibratedRed: 0.56, green: 0.38, blue: 0.28, alpha: 1.0)),
]

struct ObjSpec {
    let shape: String
    let color: String
}

struct RNG {
    private var state: UInt64
    init(seed: UInt64) {
        self.state = seed == 0 ? 0x123456789abcdef : seed
    }

    mutating func next() -> UInt64 {
        state = 6364136223846793005 &* state &+ 1442695040888963407
        return state
    }

    mutating func nextInt(_ upperBound: Int) -> Int {
        return Int(next() % UInt64(upperBound))
    }

    mutating func shuffled<T>(_ values: [T]) -> [T] {
        var arr = values
        if arr.count < 2 { return arr }
        for i in stride(from: arr.count - 1, through: 1, by: -1) {
            let j = nextInt(i + 1)
            if i != j {
                arr.swapAt(i, j)
            }
        }
        return arr
    }
}

func patchCenter(col1: Int, row1: Int) -> (Int, Int) {
    let cx = Int((Double(col1) - 0.5) * Double(patch))
    let cy = Int((Double(row1) - 0.5) * Double(patch))
    return (cx, cy)
}

func patchIndex(col1: Int, row1: Int) -> Int {
    return (row1 - 1) * gridW + (col1 - 1)
}

func drawSeparatorLines() {
    NSColor.black.setStroke()
    let path = NSBezierPath()
    path.lineWidth = 2.0
    for y in separatorLinesY {
        path.move(to: NSPoint(x: 0, y: CGFloat(canvasH - y)))
        path.line(to: NSPoint(x: CGFloat(canvasW), y: CGFloat(canvasH - y)))
    }
    path.stroke()
}

func labelFont() -> NSFont {
    if let font = NSFont(name: "Helvetica", size: labelFontSize) {
        return font
    }
    return NSFont.systemFont(ofSize: labelFontSize, weight: .regular)
}

func drawRowLabels() -> [[String: Any]] {
    let font = labelFont()
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor.black
    ]

    var meta: [[String: Any]] = []
    for (rowIdx, sym) in rowSymbols.enumerated() {
        let (_, cy) = patchCenter(col1: 1, row1: targetRows[rowIdx])
        let text = NSString(string: sym)
        let size = text.size(withAttributes: attrs)
        let x: CGFloat = 18.0
        let y = CGFloat(canvasH - cy) - size.height / 2.0
        text.draw(at: NSPoint(x: x, y: y), withAttributes: attrs)
        meta.append([
            "symbol": sym,
            "patch_index": patchIndex(col1: 1, row1: targetRows[rowIdx]),
            "row": targetRows[rowIdx],
            "grid_col": 1,
            "center_position": [14, cy],
            "glyph_size": [Int(round(size.width)), Int(round(size.height))]
        ])
    }
    return meta
}

func polygonPoints(center: CGPoint, radius: CGFloat, sides: Int, rotation: CGFloat) -> [CGPoint] {
    return (0..<sides).map { i in
        let angle = rotation + (2.0 * .pi * CGFloat(i) / CGFloat(sides))
        return CGPoint(
            x: center.x + radius * cos(angle),
            y: center.y + radius * sin(angle)
        )
    }
}

func drawShape(shape: String, color: NSColor, center: CGPoint) {
    let rect = CGRect(
        x: center.x - objectSize / 2.0,
        y: center.y - objectSize / 2.0,
        width: objectSize,
        height: objectSize
    )

    color.setFill()

    switch shape {
    case "circle":
        NSBezierPath(ovalIn: rect).fill()
    case "square":
        NSBezierPath(rect: rect).fill()
    case "triangle":
        let path = NSBezierPath()
        path.move(to: CGPoint(x: center.x, y: center.y + objectSize / 2.0))
        path.line(to: CGPoint(x: center.x + objectSize / 2.0, y: center.y - objectSize / 2.0))
        path.line(to: CGPoint(x: center.x - objectSize / 2.0, y: center.y - objectSize / 2.0))
        path.close()
        path.fill()
    case "diamond":
        let path = NSBezierPath()
        path.move(to: CGPoint(x: center.x, y: center.y + objectSize / 2.0))
        path.line(to: CGPoint(x: center.x + objectSize / 2.0, y: center.y))
        path.line(to: CGPoint(x: center.x, y: center.y - objectSize / 2.0))
        path.line(to: CGPoint(x: center.x - objectSize / 2.0, y: center.y))
        path.close()
        path.fill()
    case "pentagon":
        let pts = polygonPoints(center: center, radius: objectSize / 2.0, sides: 5, rotation: -.pi / 2.0)
        let path = NSBezierPath()
        path.move(to: pts[0])
        for pt in pts.dropFirst() { path.line(to: pt) }
        path.close()
        path.fill()
    case "hexagon":
        let pts = polygonPoints(center: center, radius: objectSize / 2.0, sides: 6, rotation: .pi / 6.0)
        let path = NSBezierPath()
        path.move(to: pts[0])
        for pt in pts.dropFirst() { path.line(to: pt) }
        path.close()
        path.fill()
    default:
        NSBezierPath(rect: rect).fill()
    }
}

func comboPool() -> [ObjSpec] {
    var out: [ObjSpec] = []
    for shape in shapeNames {
        for (name, _) in palette {
            out.append(ObjSpec(shape: shape, color: name))
        }
    }
    return out
}

func colorByName(_ name: String) -> NSColor {
    return palette.first(where: { $0.0 == name })!.1
}

func sampleRows(rng: inout RNG) -> [[ObjSpec]] {
    var remaining = comboPool()
    var rows: [[ObjSpec]] = []

    for _ in 0..<4 {
        var picked: [ObjSpec] = []
        var tries = 0
        while picked.count < 3 && tries < 1000 {
            tries += 1
            let cand = remaining[rng.nextInt(remaining.count)]
            let sameShape = picked.contains(where: { $0.shape == cand.shape })
            let sameColor = picked.contains(where: { $0.color == cand.color })
            if sameShape || sameColor { continue }
            picked.append(cand)
        }
        rows.append(picked)
        for item in picked {
            if let idx = remaining.firstIndex(where: { $0.shape == item.shape && $0.color == item.color }) {
                remaining.remove(at: idx)
            }
        }
    }
    return rows
}

func makeBitmapRep() -> NSBitmapImageRep {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: canvasW,
        pixelsHigh: canvasH,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bitmapFormat: [],
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!
    rep.size = NSSize(width: canvasW, height: canvasH)
    return rep
}

func writePNG(rep: NSBitmapImageRep, to path: URL) throws {
    let data = rep.representation(using: .png, properties: [:])!
    try data.write(to: path)
}

func jsonData(_ value: Any) throws -> Data {
    return try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
}

func argsMap() -> [String: String] {
    var out: [String: String] = [:]
    var idx = 1
    let argv = CommandLine.arguments
    while idx + 1 < argv.count {
        if argv[idx].hasPrefix("--") {
            out[argv[idx]] = argv[idx + 1]
            idx += 2
        } else {
            idx += 1
        }
    }
    return out
}

let args = argsMap()
let outDir = URL(fileURLWithPath: args["--out_dir"] ?? "synthetic_multiobject_grid_3x4", isDirectory: true)
let numImages = Int(args["--num_images"] ?? "200") ?? 200
let seed = UInt64(args["--seed"] ?? "7") ?? 7

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

var rng = RNG(seed: seed)
var records: [[String: Any]] = []
var previewReps: [NSBitmapImageRep] = []

for imageID in 0..<numImages {
    let rows = sampleRows(rng: &rng)
    let rep = makeBitmapRep()
    NSGraphicsContext.saveGraphicsState()
    let ctx = NSGraphicsContext(bitmapImageRep: rep)!
    NSGraphicsContext.current = ctx

    NSColor.white.setFill()
    NSBezierPath(rect: CGRect(x: 0, y: 0, width: canvasW, height: canvasH)).fill()
    drawSeparatorLines()
    let rowSymbolsMeta = drawRowLabels()

    var objects: [[String: Any]] = []
    var rowSummaries: [[String: Any]] = []

    for (rowIdx, rowItems) in rows.enumerated() {
        let absRow = targetRows[rowIdx]
        var rowSummaryItems: [[String: Any]] = []
        for (colIdx, spec) in rowItems.enumerated() {
            let absCol = objectCols[colIdx]
            let (cx, cy) = patchCenter(col1: absCol, row1: absRow)
            let drawPoint = CGPoint(x: CGFloat(cx), y: CGFloat(canvasH - cy))
            drawShape(shape: spec.shape, color: colorByName(spec.color), center: drawPoint)
            let x0 = Int(round(CGFloat(cx) - objectSize / 2.0))
            let y0 = Int(round(CGFloat(cy) - objectSize / 2.0))
            objects.append([
                "shape": spec.shape,
                "color": spec.color,
                "patch_index": patchIndex(col1: absCol, row1: absRow),
                "row": absRow,
                "grid_col": absCol,
                "row_index_0based": rowIdx,
                "col_index_0based": colIdx,
                "center_position": [cx, cy],
                "paste_position": [x0, y0],
                "size": Int(objectSize)
            ])
            rowSummaryItems.append([
                "shape": spec.shape,
                "color": spec.color,
                "grid_col": absCol
            ])
        }
        rowSummaries.append([
            "items": rowSummaryItems,
            "row": absRow,
            "row_index_0based": rowIdx,
            "symbol": rowSymbols[rowIdx]
        ])
    }

    NSGraphicsContext.restoreGraphicsState()

    let filename = String(format: "shapes_%03d_with_symbols.png", imageID)
    try writePNG(rep: rep, to: outDir.appendingPathComponent(filename))
    if imageID < 8 { previewReps.append(rep) }

    records.append([
        "image_id": imageID,
        "filename": filename,
        "canvas_size_x": canvasW,
        "canvas_size_y": canvasH,
        "grid_size_x": gridW,
        "grid_size_y": gridH,
        "patch_size": patch,
        "num_objects": objects.count,
        "target_rows": targetRows,
        "symbol_arrangement": rowSymbols,
        "object_cols": objectCols,
        "separator_lines_y": separatorLinesY,
        "objects": objects,
        "row_symbols": rowSymbolsMeta,
        "row_summaries": rowSummaries,
        "description": "Synthetic 4-row x 3-column grid with clean row labels and three shape objects per row."
    ])
}

try jsonData(records).write(to: outDir.appendingPathComponent("all_samples_metadata.json"))
try jsonData([
    "canvas_size_x": canvasW,
    "canvas_size_y": canvasH,
    "grid_size_x": gridW,
    "grid_size_y": gridH,
    "patch_size": patch,
    "target_rows": targetRows,
    "object_cols": objectCols,
    "symbol_arrangement": rowSymbols,
    "separator_lines_y": separatorLinesY,
    "shapes": shapeNames,
    "colors": palette.map { $0.0 },
    "num_images": numImages,
    "seed": seed
]).write(to: outDir.appendingPathComponent("dataset_config.json"))

let promptAll = """
Look at the image. The four horizontal rows are labeled on the left by symbols.
From top to bottom the row labels are @, #, $, &.
Each row contains three colored shapes between the separator lines.
Write EXACTLY four lines, one per row, using only lowercase shape words.
Format:
row @: <shape>, <shape>, <shape>
row #: <shape>, <shape>, <shape>
row $: <shape>, <shape>, <shape>
row &: <shape>, <shape>, <shape>
"""

let promptRow = """
Look at the image. The four horizontal rows are labeled on the left by symbols.
From top to bottom the row labels are @, #, $, &.
Each row contains three colored shapes between the separator lines.
What are the three shapes in the "$" row?
Answer with exactly three lowercase shape words separated by commas.
"""

try promptAll.write(to: outDir.appendingPathComponent("prompt_all_rows_shapes.txt"), atomically: true, encoding: .utf8)
try promptRow.write(to: outDir.appendingPathComponent("prompt_row_dollar_shapes.txt"), atomically: true, encoding: .utf8)

if !previewReps.isEmpty {
    let cols = 4
    let rows = 2
    let previewRep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: cols * canvasW,
        pixelsHigh: rows * canvasH,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bitmapFormat: [],
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: previewRep)
    NSColor(calibratedWhite: 0.96, alpha: 1.0).setFill()
    NSBezierPath(rect: CGRect(x: 0, y: 0, width: cols * canvasW, height: rows * canvasH)).fill()
    for (idx, rep) in previewReps.enumerated() {
        let x = (idx % cols) * canvasW
        let y = ((rows - 1) - (idx / cols)) * canvasH
        rep.draw(in: CGRect(x: x, y: y, width: canvasW, height: canvasH))
    }
    NSGraphicsContext.restoreGraphicsState()
    try writePNG(rep: previewRep, to: outDir.appendingPathComponent("preview_grid.png"))
}

print("[done] wrote dataset to \(outDir.path)")
print("[done] images=\(numImages) metadata=\(outDir.appendingPathComponent("all_samples_metadata.json").path)")
