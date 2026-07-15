#!/usr/bin/env python3
"""
Antigravity Annotation Toolbox — Backend HTTP Server
Provides API endpoints to serve video frames on-the-fly and edit/save tracking CSV files.
"""

import os
import sys
import json
import csv
import urllib.parse
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
import cv2
import numpy as np
import pandas as pd

# Add src folder to path for importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root workspace dir (btp)
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotation_tool")

# Global thread lock for OpenCV video operations
video_lock = threading.Lock()

# Global video reader cache to avoid re-opening on every frame request
class VideoReaderCache:
    def __init__(self):
        self.video_path = None
        self.cap = None

    def get_cap(self, path):
        if self.video_path == path and self.cap is not None:
            if self.cap.isOpened():
                return self.cap
            else:
                self.cap.release()

        # Open new capture
        self.cap = cv2.VideoCapture(path)
        self.video_path = path
        if not self.cap.isOpened():
            print(f"Failed to open video: {path}")
            return None
        return self.cap

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.video_path = None

VIDEO_CACHE = VideoReaderCache()

def fit_coordinate_mapper(csv_path):
    """
    Examines the original CSV contents to determine how pixels map to real-world coordinates.
    Tries to compute a homography matrix or uniform scale factor based on existing rows.
    """
    try:
        df = pd.read_csv(csv_path)
        # Filter rows with valid, non-zero pixel and world coordinates
        valid = df[(df['center_x'] > 0) & (df['center_y'] > 0) & (df['world_x'] != 0) & (df['world_y'] != 0)]
        if len(valid) < 4:
            # Try a simple scale factor fallback
            if len(valid) >= 1:
                scale_x = (valid['world_x'] / valid['center_x']).mean()
                scale_y = (valid['world_y'] / valid['center_y']).mean()
                scale = (scale_x + scale_y) / 2.0
                return {"mode": "scale", "scale": scale}
            return {"mode": "scale", "scale": 0.05} # Absolute default fallback

        # Try to fit homography
        pts_px = valid[['center_x', 'center_y']].values.astype(np.float32)
        pts_world = valid[['world_x', 'world_y']].values.astype(np.float32)

        # Drop duplicate points to prevent singular matrices
        _, idx = np.unique(pts_px, axis=0, return_index=True)
        pts_px = pts_px[idx]
        pts_world = pts_world[idx]

        if len(pts_px) >= 4:
            H, mask = cv2.findHomography(pts_px, pts_world, cv2.RANSAC, 5.0)
            if H is not None:
                return {"mode": "homography", "H": H}

        # Fallback to scale factor
        scale_x = (valid['world_x'] / valid['center_x']).mean()
        scale_y = (valid['world_y'] / valid['center_y']).mean()
        scale = (scale_x + scale_y) / 2.0
        return {"mode": "scale", "scale": scale}

    except Exception as e:
        print(f"Error fitting coordinate mapper: {e}")
        return {"mode": "scale", "scale": 0.05}

def apply_coordinate_mapper(mapper, cx, cy):
    """Maps pixel coordinates to world coordinates based on the fitted mapper."""
    if mapper["mode"] == "homography":
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, mapper["H"])
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])
    else:
        scale = mapper["scale"]
        return cx * scale, cy * scale

def extract_frames_bg(video_path, frame_dir):
    try:
        print(f"[BG Extractions] Starting background frame extraction for: {video_path} -> {frame_dir}")
        os.makedirs(frame_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[BG Extractions] Failed to open video: {video_path}")
            return
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            frame_idx += 1
            if frame_idx % 10 == 0:
                import time
                time.sleep(0.005) # Yield thread
                
        cap.release()
        print(f"[BG Extractions] Completed. Extracted {frame_idx} frames successfully.")
    except Exception as e:
        print(f"[BG Extractions] Error: {e}")

def trigger_frame_extraction(video_rel_path):
    try:
        video_abs_path = os.path.join(WORKSPACE_DIR, video_rel_path)
        video_name = os.path.splitext(os.path.basename(video_rel_path))[0]
        frame_dir = os.path.join(WORKSPACE_DIR, "data", "frames", video_name)
        
        # Check if already extracted
        if os.path.exists(frame_dir):
            existing_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
            cap = cv2.VideoCapture(video_abs_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if existing_frames >= total_frames - 5:
                print(f"[BG Extractions] Frames already extracted ({existing_frames}/{total_frames}). Skipping.")
                return
                
        # Start background thread
        threading.Thread(target=extract_frames_bg, args=(video_abs_path, frame_dir), daemon=True).start()
    except Exception as e:
        print(f"[BG Extractions] Failed to trigger extraction: {e}")


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow cross-origin requests for testing
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # API Endpoints
        if path == "/api/videos":
            self.handle_list_videos()
        elif path == "/api/csvs":
            self.handle_list_csvs()
        elif path == "/api/video-details":
            self.handle_video_details(query)
        elif path == "/api/frame":
            self.handle_get_frame(query)
        elif path == "/api/csv-data":
            self.handle_get_csv_data(query)
        else:
            # Serve static frontend files
            self.handle_serve_static(path)

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/save-csv":
            self.handle_save_csv(query)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_serve_static(self, path):
        # Default to index.html
        if path == "/" or path == "":
            file_path = os.path.join(STATIC_DIR, "index.html")
        else:
            # Strip leading slash
            file_path = os.path.join(STATIC_DIR, path.lstrip("/"))

        # Prevent directory traversal attacks
        if not os.path.abspath(file_path).startswith(os.path.abspath(STATIC_DIR)):
            self.send_error(403, "Access denied")
            return

        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, "File not found")
            return

        # Determine Content-Type
        content_type = "text/plain"
        if file_path.endswith(".html"):
            content_type = "text/html"
        elif file_path.endswith(".css"):
            content_type = "text/css"
        elif file_path.endswith(".js"):
            content_type = "text/javascript"
        elif file_path.endswith(".png"):
            content_type = "image/png"
        elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif file_path.endswith(".ico"):
            content_type = "image/x-icon"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.write_in_chunks(content)
        except Exception as e:
            self.send_error(500, f"Internal server error: {e}")

    def handle_list_videos(self):
        """Scans data/video/ and outputs/video/ for .mp4 files."""
        videos = []
        # Check data/video/
        data_video_dir = os.path.join(WORKSPACE_DIR, "data", "video")
        if os.path.exists(data_video_dir):
            for f in os.listdir(data_video_dir):
                if f.endswith(".mp4"):
                    videos.append(os.path.join("data", "video", f).replace("\\", "/"))
        # Check outputs/video/
        outputs_video_dir = os.path.join(WORKSPACE_DIR, "outputs", "video")
        if os.path.exists(outputs_video_dir):
            for f in os.listdir(outputs_video_dir):
                if f.endswith(".mp4"):
                    videos.append(os.path.join("outputs", "video", f).replace("\\", "/"))

        response_bytes = json.dumps(videos).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_list_csvs(self):
        """Scans src/outputs/csv/ and outputs/csv/ for .csv files."""
        csvs = []
        # Check src/outputs/csv/
        src_csv_dir = os.path.join(WORKSPACE_DIR, "src", "outputs", "csv")
        if os.path.exists(src_csv_dir):
            for f in os.listdir(src_csv_dir):
                if f.endswith(".csv"):
                    csvs.append(os.path.join("src", "outputs", "csv", f).replace("\\", "/"))
        # Check outputs/csv/
        outputs_csv_dir = os.path.join(WORKSPACE_DIR, "outputs", "csv")
        if os.path.exists(outputs_csv_dir):
            for f in os.listdir(outputs_csv_dir):
                if f.endswith(".csv"):
                    csvs.append(os.path.join("outputs", "csv", f).replace("\\", "/"))

        response_bytes = json.dumps(csvs).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_video_details(self, query):
        video_rel_path = query.get("video", [None])[0]
        if not video_rel_path:
            self.send_error(400, "Missing 'video' parameter")
            return

        # Trigger background frame extraction to eliminate seek lag
        trigger_frame_extraction(video_rel_path)

        video_abs_path = os.path.join(WORKSPACE_DIR, video_rel_path)
        with video_lock:
            cap = VIDEO_CACHE.get_cap(video_abs_path)
            if cap is None:
                self.send_error(400, "Could not open video file")
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0 # Default fallback

        details = {
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": fps
        }

        response_bytes = json.dumps(details).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def handle_get_frame(self, query):
        video_rel_path = query.get("video", [None])[0]
        frame_idx_str = query.get("frame", [None])[0]

        if not video_rel_path or frame_idx_str is None:
            self.send_error(400, "Missing video or frame parameter")
            return

        try:
            frame_idx = int(frame_idx_str)
        except ValueError:
            self.send_error(400, "Invalid frame index")
            return

        # Check if pre-extracted frame file is available on disk
        video_name = os.path.splitext(os.path.basename(video_rel_path))[0]
        frame_path = os.path.join(WORKSPACE_DIR, "data", "frames", video_name, f"frame_{frame_idx:06d}.jpg")
        if os.path.exists(frame_path):
            try:
                with open(frame_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.write_in_chunks(content)
                return
            except Exception as e:
                print(f"[Server] Failed to read pre-extracted frame: {e}. Falling back to OpenCV seek.")

        # Fallback to OpenCV seek if frame is not yet extracted to disk
        video_abs_path = os.path.join(WORKSPACE_DIR, video_rel_path)
        with video_lock:
            cap = VIDEO_CACHE.get_cap(video_abs_path)
            if cap is None:
                self.send_error(400, "Could not open video file")
                return

            # Seek and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Return a blank dark placeholder frame if seek fails (e.g. beyond end of video)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Encode to JPEG
            ret_enc, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not ret_enc:
            self.send_error(500, "Failed to encode frame")
            return

        content = jpeg_bytes.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.write_in_chunks(content)

    def handle_get_csv_data(self, query):
        csv_rel_path = query.get("csv", [None])[0]
        if not csv_rel_path:
            self.send_error(400, "Missing 'csv' parameter")
            return

        csv_abs_path = os.path.join(WORKSPACE_DIR, csv_rel_path)
        if not os.path.exists(csv_abs_path):
            self.send_error(404, "CSV file not found")
            return

        try:
            # Read CSV and format as JSON
            df = pd.read_csv(csv_abs_path)
            # Ensure columns exist
            for col in ["frame", "track_id", "class_name", "x1", "y1", "x2", "y2", "center_x", "center_y", "world_x", "world_y", "confidence", "velocity_ms"]:
                if col not in df.columns:
                    if col == "confidence":
                        df["confidence"] = 1.0
                    elif col == "velocity_ms":
                        df["velocity_ms"] = 0.0
                    else:
                        df[col] = 0.0

            # Convert to dictionary indexed by frame index
            records = df.to_dict(orient="records")
            data_by_frame = {}
            for row in records:
                f_idx = int(row["frame"])
                if f_idx not in data_by_frame:
                    data_by_frame[f_idx] = []
                # Clean NaNs
                for k, v in row.items():
                    if isinstance(v, float) and np.isnan(v):
                        row[k] = 0.0
                data_by_frame[f_idx].append(row)

            response_bytes = json.dumps(data_by_frame).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        except Exception as e:
            self.send_error(500, f"Failed to parse CSV: {e}")

    def handle_save_csv(self, query):
        csv_rel_path = query.get("csv", [None])[0]
        if not csv_rel_path:
            self.send_error(400, "Missing 'csv' parameter")
            return

        csv_abs_path = os.path.join(WORKSPACE_DIR, csv_rel_path)
        if not os.path.exists(csv_abs_path):
            self.send_error(404, "CSV file not found")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8"))
            edited_frames = body.get("frames", {})
            fps = float(body.get("fps", 30.0))

            # 1. Fit coordinate mapper from original CSV file
            mapper = fit_coordinate_mapper(csv_abs_path)
            print(f"Fitted mapper: {mapper}")

            # 2. Re-compile all records
            all_records = []
            for f_idx_str, boxes in edited_frames.items():
                f_idx = int(f_idx_str)
                for box in boxes:
                    x1 = float(box["x1"])
                    y1 = float(box["y1"])
                    x2 = float(box["x2"])
                    y2 = float(box["y2"])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # Map coordinates to world
                    wx, wy = apply_coordinate_mapper(mapper, cx, cy)

                    all_records.append({
                        "frame": f_idx,
                        "track_id": int(box["track_id"]),
                        "class_name": str(box["class_name"]),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "center_x": round(cx, 2),
                        "center_y": round(cy, 2),
                        "world_x": round(wx, 4),
                        "world_y": round(wy, 4),
                        "confidence": round(float(box.get("confidence", 1.0)), 4),
                        "velocity_ms": 0.0 # Will calculate in step 3
                    })

            # Create dataframe
            df_new = pd.DataFrame(all_records)
            if len(df_new) > 0:
                # Sort by track_id and frame for velocity calculation
                df_new = df_new.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

                # 3. Recalculate velocities per track
                prev_track_id = None
                prev_wx, prev_wy = None, None
                prev_frame = None

                velocities = []
                for idx, row in df_new.iterrows():
                    tid = row["track_id"]
                    frame = row["frame"]
                    wx = row["world_x"]
                    wy = row["world_y"]

                    vel_ms = 0.0
                    if tid == prev_track_id and prev_frame is not None:
                        dt = frame - prev_frame
                        if dt > 0:
                            dist = np.hypot(wx - prev_wx, wy - prev_wy)
                            vel_ms = (dist * fps) / dt
                    
                    velocities.append(round(float(vel_ms), 4))
                    
                    prev_track_id = tid
                    prev_wx, prev_wy = wx, wy
                    prev_frame = frame

                df_new["velocity_ms"] = velocities

                # Re-sort back by frame and track_id for standard layout
                df_new = df_new.sort_values(by=["frame", "track_id"]).reset_index(drop=True)

            # Write back to CSV
            df_new.to_csv(csv_abs_path, index=False)
            print(f"Saved {len(df_new)} records to CSV: {csv_abs_path}")

            response = {"status": "success", "rows_written": len(df_new)}
            response_bytes = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_error(500, f"Failed to save CSV: {e}")

    def write_in_chunks(self, data, chunk_size=128 * 1024):
        """Write response body in chunks to avoid blocking issues with huge files."""
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            self.wfile.write(chunk)


def run_server():
    # Make sure static directory exists
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = ('', PORT)
    # ThreadingTCPServer handles concurrent requests (e.g. frame fetching while scrolling)
    httpd = ThreadingTCPServer(server_address, AnnotationRequestHandler)
    print(f"Antigravity Annotation Toolbox Server running on http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        VIDEO_CACHE.close()
        httpd.server_close()

if __name__ == "__main__":
    run_server()
