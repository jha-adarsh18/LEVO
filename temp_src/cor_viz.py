import cv2
import argparse
import numpy as np

def visualize_matches(image, matches):
    """
    Visualize matches between two copies of the same image with pixel gap (stacked vertically)
    """
    h, w, _ = image.shape
    gap = 10  # 10 pixel gap between images
    
    # Create vertical canvas with gap
    canvas = np.zeros((h * 2 + gap, w, 3), dtype=np.uint8)
    
    # Place images vertically with gap
    offset_top = 0
    offset_bottom = h + gap
    canvas[offset_top:offset_top+h, :] = image
    canvas[offset_bottom:offset_bottom+h, :] = image
    
    # Draw white horizontal line as separator
    separator_color = (255, 255, 255)
    sep_y = h + gap // 2
    cv2.line(canvas, (0, sep_y), (w-1, sep_y), separator_color, 2)
    
    # Draw matches in green with doubled thickness
    color = (0, 255, 0)
    for kp1, kp2 in matches:
        pt1 = (int(kp1[0]), int(kp1[1]) + offset_top)
        pt2 = (int(kp2[0]), int(kp2[1]) + offset_bottom)
        cv2.line(canvas, pt1, pt2, color, 2, cv2.LINE_AA)  # Changed thickness from 1 to 2
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, cv2.LINE_AA)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--output', type=str, default='matches_visualization.png',
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Load the image
    img = cv2.imread(args.image_path)
    
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    print(f"Loaded image: {args.image_path}")
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Define matches (left image coords -> right image coords)
    matches = [
        ((432, 234), (432, 234)),
        ((417, 247), (417, 247)),
        ((291, 337), (291, 337)), 
        ((426, 349), (426, 349)),
        ((415, 340), (415, 340)),
        ((433, 445), (433, 445)),
        ((464, 403), (464, 403)),
        ((463, 429), (463, 429)),
        ((377, 335), (377, 335)), 
        ((401, 366), (401, 366))
    ]
    
    print(f"Visualizing {len(matches)} matches...")
    
    # Create visualization
    viz = visualize_matches(img, matches)
    
    # Save output
    cv2.imwrite(args.output, viz)
    print(f"Saved visualization to {args.output}")
    
    # Also display it
    cv2.namedWindow("Matches Visualization")
    cv2.imshow("Matches Visualization", viz)
    print("\nPress any key to close window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()