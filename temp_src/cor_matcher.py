import cv2
import argparse
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        param['x'] = x
        param['y'] = y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the image file')
    
    args = parser.parse_args()
    
    # Load the image
    img = cv2.imread(args.image_path)
    
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"Loaded image: {args.image_path}")
    print(f"Image size: {w}x{h}")
    print("\nHover mouse to see coordinates. Press any key to close.")
    
    # Create window
    window_name = "Image Viewer - Hover for coordinates"
    cv2.namedWindow(window_name)
    
    # Setup mouse callback
    mouse_data = {'x': 0, 'y': 0}
    cv2.setMouseCallback(window_name, mouse_callback, mouse_data)
    
    # Display loop
    while True:
        display = img.copy()
        x, y = mouse_data['x'], mouse_data['y']
        
        # Draw coordinate text at top
        text = f"X: {x}  Y: {y}"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw crosshair at cursor position
        if 0 <= x < w and 0 <= y < h:
            cv2.line(display, (x, 0), (x, h-1), (0, 255, 0), 1)
            cv2.line(display, (0, y), (w-1, y), (0, 255, 0), 1)
        
        cv2.imshow(window_name, display)
        
        if cv2.waitKey(1) != -1:
            break
    
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == '__main__':
    main()