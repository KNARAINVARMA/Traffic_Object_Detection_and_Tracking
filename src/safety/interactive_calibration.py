import cv2
import math
import sys
from pathlib import Path

# Global state for click callback
points = []

def click_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        scale_display = param.get('scale_display', 1.0)
        points.append((x, y))
        
        orig_x = x / scale_display
        orig_y = y / scale_display
        print(f"Point {len(points)}: Display(X={x}, Y={y}) -> Original(X={orig_x:.2f}, Y={orig_y:.2f})")
        
        # Draw a circle on the clicked point
        cv2.circle(param['img'], (x, y), 5, (0, 0, 255), -1)
        if len(points) == 2:
            # Draw line between points
            cv2.line(param['img'], points[0], points[1], (0, 255, 0), 2)
            
            # Compute Euclidean distance in display coordinates
            x1, y1 = points[0]
            x2, y2 = points[1]
            dist_display = math.hypot(x2 - x1, y2 - y1)
            
            # Compute Euclidean distance in original coordinates
            orig_x1 = x1 / scale_display
            orig_y1 = y1 / scale_display
            orig_x2 = x2 / scale_display
            orig_y2 = y2 / scale_display
            dist_original = math.hypot(orig_x2 - orig_x1, orig_y2 - orig_y1)
            
            print(f"\nDisplay Distance  : {dist_display:.2f} px")
            print(f"Original Distance : {dist_original:.2f} px")
            print(f"Display Scale     : {scale_display:.6f}")
            
            # Compute scale factor for 7.0m lane width
            lane_width_m = 7.0
            scale = lane_width_m / dist_original
            print(f"Calculated Scale Factor: {scale:.6f} meters/pixel")
            print(f"\nSuggested command arguments:")
            print(f"  --lane-width-px {dist_original:.2f} --lane-width-m 7.0")
        cv2.imshow("Interactive Calibration Tool", param['img'])

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_calibration.py <path_to_video>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read first frame from video '{video_path}'")
        sys.exit(1)
        
    # Resize display if image is too large for screen
    h, w = frame.shape[:2]
    max_dim = 1000
    scale_display = 1.0
    if max(h, w) > max_dim:
        scale_display = max_dim / max(h, w)
        frame_display = cv2.resize(frame, (int(w * scale_display), int(h * scale_display)))
    else:
        frame_display = frame.copy()
        
    print("=== Interactive Lane Calibration Tool ===")
    print("1. Click the start of the lane boundary (e.g. inner curb).")
    print("2. Click the end of the lane boundary (e.g. outer lane separator line).")
    print("Note: Measure along a radial line perpendicular to the lane trajectory for maximum accuracy.")
    print("Press 'r' to reset points, or 'q' to quit.")
    
    img_display = frame_display.copy()
    cv2.namedWindow("Interactive Calibration Tool")
    cv2.setMouseCallback("Interactive Calibration Tool", click_callback, {
        'img': img_display,
        'scale_display': scale_display
    })
    
    while True:
        cv2.imshow("Interactive Calibration Tool", img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            points.clear()
            img_display = frame_display.copy()
            cv2.setMouseCallback("Interactive Calibration Tool", click_callback, {
                'img': img_display,
                'scale_display': scale_display
            })
            print("\nReset points. Click again.")
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
