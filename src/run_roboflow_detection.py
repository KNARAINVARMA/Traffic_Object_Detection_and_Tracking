"""
Roboflow-based traffic detection and tracking (single-colour, full-frame).

Detects vehicles using a Roboflow model, tracks them with ByteTrack,
and annotates the entire video with boxes, labels, and traces in one colour.

Usage (from traffic_tracking/):
    export ROBOFLOW_API_KEY=your_key_here
    python src/run_roboflow_detection.py

Or with custom paths:
    python src/run_roboflow_detection.py \\
        --source_video_path /path/to/video.mp4 \\
        --target_video_path outputs/video/tracked.mp4
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model
from tqdm import tqdm

import supervision as sv

COLOR = sv.Color.from_hex("#3C76D1")


class VideoProcessor:
    def __init__(
        self,
        roboflow_api_key: str,
        model_id: str,
        source_video_path: str,
        target_video_path: str | None = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.box_annotator = sv.BoxAnnotator(color=sv.ColorPalette([COLOR]))
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette([COLOR]), text_color=sv.Color.WHITE
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=sv.ColorPalette([COLOR]),
            position=sv.Position.CENTER,
            trace_length=100,
            thickness=2,
        )

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            Path(self.target_video_path).parent.mkdir(parents=True, exist_ok=True)
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels
        )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model.infer(
            frame, confidence=self.conf_threshold, iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(results)
        detections.class_id = np.zeros(len(detections), dtype=int)
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame, detections)


def _default_paths():
    """Default source and output paths."""
    root = Path(__file__).resolve().parent.parent
    default_video = root.parent / "DJI_20250124120127_0005_D.MP4"
    return (
        str(default_video),
        str(root / "outputs" / "video" / "roboflow_DJI_tracked_video_0005.mp4"),
    )


def main(
    source_video_path: str | None = None,
    target_video_path: str | None = None,
    roboflow_api_key: str | None = None,
    model_id: str = "vehicle-count-in-drone-video/6",
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> None:
    """
    Traffic detection & tracking with Roboflow + supervision.

    Args:
        source_video_path: Path to the source video.
        target_video_path: Path for the output video.
        roboflow_api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var).
        model_id: Roboflow model ID.
        confidence_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
    """
    default_src, default_dst = _default_paths()
    source_video_path = source_video_path or default_src
    target_video_path = target_video_path or default_dst

    api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Roboflow API key is required. Set ROBOFLOW_API_KEY or pass roboflow_api_key."
        )

    if not Path(source_video_path).exists():
        raise FileNotFoundError(
            f"Source video not found: {source_video_path}. "
            "Place your video there or pass --source_video_path."
        )

    processor = VideoProcessor(
        roboflow_api_key=api_key,
        model_id=model_id,
        source_video_path=source_video_path,
        target_video_path=target_video_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    processor.process_video()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Roboflow vehicle detection and tracking on a video."
    )
    parser.add_argument(
        "--source_video_path", "-i",
        default=None,
        help="Input video path.",
    )
    parser.add_argument(
        "--target_video_path", "-o",
        default=None,
        help="Output video path.",
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY).",
    )
    parser.add_argument(
        "--model_id",
        default="vehicle-count-in-drone-video/6",
        help="Roboflow model ID.",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        roboflow_api_key=args.roboflow_api_key,
        model_id=args.model_id,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
