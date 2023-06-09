import cv2
import numpy as np

def detect_forgery(original_image_path, forged_image_path, block_size=16, threshold=0.99):
    # Load the original and forged images
    original_image = cv2.imread(original_image_path)
    forged_image = cv2.imread(forged_image_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    forged_gray = cv2.cvtColor(forged_image, cv2.COLOR_BGR2GRAY)

    # Divide images into blocks
    original_blocks = []
    forged_blocks = []
    for i in range(0, original_gray.shape[0] - block_size, block_size):
        for j in range(0, original_gray.shape[1] - block_size, block_size):
            original_block = original_gray[i:i+block_size, j:j+block_size]
            forged_block = forged_gray[i:i+block_size, j:j+block_size]
            original_blocks.append(original_block)
            forged_blocks.append(forged_block)

    # Compute block-wise absolute difference and compare
    diff_scores = []
    for original_block, forged_block in zip(original_blocks, forged_blocks):
        diff = cv2.absdiff(original_block, forged_block)
        diff_score = 1.0 - (np.mean(diff) / 255.0)
        diff_scores.append(diff_score)

    # Identify forged blocks based on the difference scores
    forged_blocks_indices = [i for i, score in enumerate(diff_scores) if score < threshold]

    # Create a copy of the forged image for drawing the detected regions
    result = forged_image.copy()

    # Draw rectangles around the detected forged regions
    for index in forged_blocks_indices:
        x = (index % (original_gray.shape[1] // block_size)) * block_size
        y = (index // (original_gray.shape[1] // block_size)) * block_size
        cv2.rectangle(result, (x, y), (x + block_size, y + block_size), (0, 0, 255), 2)

    # Display the result
    # let's downscale the image using new  width and height
    down_width = 500    
    down_height = 500
    down_points = (down_width, down_height)
    img1 = cv2.resize(result, down_points, interpolation= cv2.INTER_LINEAR)
    img2 = cv2.resize(original_image, down_points, interpolation= cv2.INTER_LINEAR)
    cv2.imshow("Original", img2)
    cv2.imshow('Result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
detect_forgery('./images/original-sample.jpg', './images/forged-sample-1.jpg', block_size=90, threshold=0.99)