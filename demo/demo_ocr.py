
import argparse
import json
import random
from pathlib import Path

import cv2
import pandas as pd
import torch
import easyocr
from ultralytics import YOLO


def run_ocr_on_image(image_path, detector, reader, output_dir, conf_threshold=0.25, imgsz=640):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Не удалось открыть изображение: {image_path}")
        return None

    original = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    results = detector(
        str(image_path),
        conf=conf_threshold,
        imgsz=imgsz,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )[0]

    detections = []

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = original[y1:y2, x1:x2]

            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ocr_result = reader.readtext(crop_rgb, detail=1, paragraph=False)

            recognized_text = ""
            ocr_confidence = 0.0

            if len(ocr_result) > 0:
                best = max(ocr_result, key=lambda x: x[2])
                recognized_text = best[1]
                ocr_confidence = float(best[2])

            detections.append({
                "image": image_path.name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "det_confidence": float(score),
                "recognized_text": recognized_text,
                "ocr_confidence": ocr_confidence
            })

            label = recognized_text if recognized_text else "text"
            label = label[:30]

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_bgr,
                f"{label} {score:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    output_image_path = output_dir / f"ocr_result_{image_path.stem}.jpg"
    cv2.imwrite(str(output_image_path), image_bgr)

    return {
        "image": str(image_path),
        "output_image": str(output_image_path),
        "num_detected_text_regions": len(detections),
        "detections": detections
    }


def collect_input_images(input_path, default_val_txt, num_samples):
    input_path = Path(input_path) if input_path else None

    if input_path and input_path.is_file():
        return [input_path]

    if input_path and input_path.is_dir():
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(input_path.glob(ext))
        return images

    with open(default_val_txt, "r", encoding="utf-8") as f:
        val_images = [Path(line.strip()) for line in f.read().splitlines() if line.strip()]

    random.seed(42)
    return random.sample(val_images, min(num_samples, len(val_images)))


def main():
    parser = argparse.ArgumentParser(description="Demo OCR module: text detection + recognition")
    parser.add_argument("--input", type=str, default=None, help="Path to image or folder with images")
    parser.add_argument("--output", type=str, default=None, help="Output folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of validation images if input is not specified")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]

    weights_path = project_dir / "models" / "yolo11n_text_detection" / "weights" / "best.pt"
    val_txt = project_dir / "data" / "raw" / "coco_text_v2" / "archive" / "val.txt"

    output_dir = Path(args.output) if args.output else project_dir / "results" / "demo_ocr_script"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Project dir:", project_dir)
    print("Weights:", weights_path)
    print("Weights exists:", weights_path.exists())
    print("Output dir:", output_dir)

    detector = YOLO(str(weights_path))

    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )

    input_images = collect_input_images(args.input, val_txt, args.num_samples)
    print("Images for processing:", len(input_images))

    all_results = []

    for image_path in input_images:
        result = run_ocr_on_image(
            image_path=image_path,
            detector=detector,
            reader=reader,
            output_dir=output_dir,
            conf_threshold=args.conf,
            imgsz=640
        )

        if result is not None:
            all_results.append(result)
            print(f"{Path(image_path).name}: text regions = {result['num_detected_text_regions']}")

    json_path = output_dir / "demo_ocr_script_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    rows = []

    for item in all_results:
        for det in item["detections"]:
            rows.append({
                "image": det["image"],
                "bbox": det["bbox"],
                "det_confidence": det["det_confidence"],
                "recognized_text": det["recognized_text"],
                "ocr_confidence": det["ocr_confidence"]
            })

    excel_path = output_dir / "demo_ocr_script_results.xlsx"
    pd.DataFrame(rows).to_excel(excel_path, index=False)

    print("JSON saved:", json_path)
    print("Excel saved:", excel_path)
    print("Annotated images saved:", output_dir)


if __name__ == "__main__":
    main()
