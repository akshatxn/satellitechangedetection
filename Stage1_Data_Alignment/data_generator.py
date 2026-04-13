import cv2
import numpy as np
import os

def generate_shifted_data():
    # 1. Create a base image (Time 1) with some shapes
    img1 = np.ones((512, 512, 3), dtype=np.uint8) * 100 # Gray background
    
    # Draw a rectangle and a circle (Landmarks)
    cv2.rectangle(img1, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img1, (400, 400), 50, (0, 0, 255), -1)
    
    cv2.imwrite("test_t1.jpg", img1)

    # 2. Create Time 2 (Shifted version of Time 1)
    rows, cols = img1.shape[:2]
    
    # Translation Matrix: Shift x by 30 pixels, y by 20 pixels
    M = np.float32([[1, 0, 30], [0, 1, 20]]) 
    img2 = cv2.warpAffine(img1, M, (cols, rows))
    
    # Add a change (a new building) so it's not identical
    cv2.rectangle(img2, (150, 150), (200, 200), (0, 255, 0), -1) 

    cv2.imwrite("test_t2.jpg", img2)
    print("✅ Created test_t1.jpg and test_t2.jpg (Shifted)")

if __name__ == "__main__":
    generate_shifted_data()