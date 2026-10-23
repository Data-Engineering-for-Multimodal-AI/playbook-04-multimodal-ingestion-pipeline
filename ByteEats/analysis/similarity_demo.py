#!/usr/bin/env python3
"""
ByteEats Image Similarity Demo
-----------------------------
Demonstrates how vector embeddings can be used to compare food images
and find similar dishes using various techniques.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pickle
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Configuration
# Sample data ships in ByteEats/data, found relative to this file so the
# script works from any folder. Outputs go to the current folder.
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
REFERENCE_IMAGES_DIR = os.path.join(DATA_DIR, "reference_images")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_bowls")
RESULTS_DIR = "similarity_results"

# Ensure directories exist
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_resize_image(image_path, target_size=(224, 224)):
    """Load and resize image using OpenCV"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_resized = cv2.resize(img, target_size)
    return img, img_resized

def extract_color_histograms(image):
    """Extract color histograms from an image"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract histograms
    hist_bgr = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist_lab = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(hist_bgr, hist_bgr, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_lab, hist_lab, 0, 1, cv2.NORM_MINMAX)
    
    # Flatten and combine
    hist_bgr = hist_bgr.flatten()
    hist_hsv = hist_hsv.flatten()
    hist_lab = hist_lab.flatten()
    
    return np.concatenate([hist_bgr, hist_hsv, hist_lab])

def extract_orb_features(image, max_features=100):
    """Extract ORB features from an image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        # If no features detected, return zero vector
        return np.zeros(max_features * 32, dtype=np.float32)
    
    # Ensure we have a fixed-size feature vector
    if len(keypoints) < max_features:
        # Pad with zeros
        padded = np.zeros((max_features, 32), dtype=np.uint8)
        padded[:len(keypoints)] = descriptors
        descriptors = padded
    elif len(keypoints) > max_features:
        # Truncate
        descriptors = descriptors[:max_features]
    
    # Flatten and convert to float
    return descriptors.flatten().astype(np.float32)

def extract_edge_features(image):
    """Extract edge-based features from an image"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Compute edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Extract edge direction histogram
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and angle
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
    
    # Create angle histogram (36 bins of 10 degrees each)
    angle_hist, _ = np.histogram(angle[magnitude > 50], bins=36, range=(0, 360))
    angle_hist = angle_hist / np.sum(angle_hist) if np.sum(angle_hist) > 0 else angle_hist
    
    # Return combined features
    return np.concatenate([[edge_density], angle_hist])

def get_opencv_embeddings(image_paths, target_size=(224, 224)):
    """Generate embeddings using OpenCV features"""
    embeddings = []
    original_images = []
    all_features = []
    
    for path in image_paths:
        # Load and resize image
        original, resized = load_and_resize_image(path, target_size)
        original_images.append(original)
        
        # Extract various features
        color_hist = extract_color_histograms(resized)
        edge_feats = extract_edge_features(resized)
        orb_feats = extract_orb_features(resized)
        
        # Combine all features
        combined_features = np.concatenate([
            color_hist,
            edge_feats,
            # Don't use ORB features directly as they're binary and too sparse
            # Instead keep track separately for custom similarity
            # orb_feats
        ])
        
        all_features.append({
            'combined': combined_features,
            'color_hist': color_hist,
            'edge_feats': edge_feats,
            'orb_feats': orb_feats
        })
        
        # Use PCA to reduce dimensionality
        embeddings.append(combined_features)
    
    # If we have enough samples, apply dimensionality reduction
    if len(embeddings) > 2:
        # Stack all embeddings
        stacked = np.vstack(embeddings)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(len(embeddings) - 1, 100))
        reduced = pca.fit_transform(stacked)
        
        # Replace embeddings with reduced versions
        embeddings = [reduced[i] for i in range(len(embeddings))]
    
    return embeddings, original_images, all_features

def calculate_similarity_matrix(embeddings, feature_data=None):
    """Calculate cosine similarity matrix between embeddings"""
    if feature_data is None:
        # Standard cosine similarity for embeddings
        return cosine_similarity(embeddings)
    
    # Calculate a custom similarity matrix incorporating different feature types
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Skip self
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            
            # Calculate similarity components
            color_similarity = cosine_similarity(
                feature_data[i]['color_hist'].reshape(1, -1),
                feature_data[j]['color_hist'].reshape(1, -1)
            )[0, 0]
            
            edge_similarity = cosine_similarity(
                feature_data[i]['edge_feats'].reshape(1, -1),
                feature_data[j]['edge_feats'].reshape(1, -1)
            )[0, 0]
            
            # For ORB binary features, use Hamming distance
            orb_i = feature_data[i]['orb_feats'].reshape(-1, 32).astype(np.uint8)
            orb_j = feature_data[j]['orb_feats'].reshape(-1, 32).astype(np.uint8)
            
            # Count matching descriptors
            matching_count = 0
            for di in orb_i:
                if np.any(di):  # Skip zero descriptors
                    min_dist = np.min([cv2.norm(di, dj, cv2.NORM_HAMMING) for dj in orb_j if np.any(dj)])
                    if min_dist < 50:  # Threshold for matching
                        matching_count += 1
            
            max_possible = min(np.sum(np.any(orb_i, axis=1)), np.sum(np.any(orb_j, axis=1)))
            orb_similarity = matching_count / max_possible if max_possible > 0 else 0
            
            # Weighted combination
            similarity_matrix[i, j] = 0.5 * color_similarity + 0.3 * edge_similarity + 0.2 * orb_similarity
    
    return similarity_matrix

def visualize_similarity_matrix(similarity_matrix, labels, output_path):
    """Visualize similarity matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Similarity Score')
    
    # Add labels
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    # Add values in cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="w")
    
    plt.title("Food Image Similarity Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_embeddings_tsne(embeddings, labels, images, output_path):
    """Visualize embeddings using t-SNE with image thumbnails"""
    if len(embeddings) < 2:
        print("Need at least 2 images for t-SNE visualization")
        return
    
    # Convert embeddings to numpy array if it's a list
    embeddings_array = np.array(embeddings)
    
    # Reduce dimensions with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(len(embeddings)-1, 1)))
    embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0)
    
    # Add thumbnail images
    for i, (x, y) in enumerate(embeddings_2d):
        # Convert image to thumbnail
        img = images[i]
        # Resize image for display
        h, w = img.shape[:2]
        max_size = 100
        scale = min(max_size/h, max_size/w)
        thumbnail = cv2.resize(img, (int(w*scale), int(h*scale)))
        # Convert BGR to RGB for matplotlib
        thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        
        # Add image as thumbnail
        imagebox = plt.imshow(thumbnail, extent=(x-20, x+20, y-20, y+20), 
                             aspect='auto', zorder=1000)
        imagebox.set_clim([0,1])
        
        # Add label below image
        plt.text(x, y-25, labels[i], ha='center', fontsize=9)
    
    plt.title("Food Image Embedding Visualization (t-SNE)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def find_most_similar(query_idx, similarity_matrix, labels):
    """Find most similar images to a query image"""
    similarities = similarity_matrix[query_idx]
    # Sort by similarity (descending)
    most_similar_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in most_similar_indices:
        if idx == query_idx:
            continue  # Skip self
        results.append({
            "index": int(idx),
            "label": labels[idx],
            "similarity": similarities[idx]
        })
    
    return results

def visualize_query_results(query_image, similar_images, similarities, labels, output_path):
    """Visualize query image and its most similar matches"""
    n_similar = len(similar_images)
    
    plt.figure(figsize=(12, 4))
    
    # Plot query image
    plt.subplot(1, n_similar + 1, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title("Query Image")
    plt.axis('off')
    
    # Plot similar images
    for i, (img, sim) in enumerate(zip(similar_images, similarities)):
        plt.subplot(1, n_similar + 1, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Sim: {sim:.2f}\n{labels[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Image Similarity Demo")
    parser.add_argument('--method', choices=['color', 'edge', 'combined'], default='combined',
                        help='Feature extraction method to use')
    args = parser.parse_args()
    
    # Get image paths
    reference_image_path = os.path.join(REFERENCE_IMAGES_DIR, "salmon_poke_bowl.jpg")
    
    # Find any available sample images
    sample_paths = list(Path(SAMPLE_IMAGES_DIR).glob("*.jpg")) + list(Path(SAMPLE_IMAGES_DIR).glob("*.png"))
    
    # Add video frame from extracted features
    video_frame_path = "video_features_enhanced/delivery-packaging-7/frame_11.jpg"
    
    # Combine all image paths
    all_image_paths = [reference_image_path] + [str(p) for p in sample_paths]
    if os.path.exists(video_frame_path):
        all_image_paths.append(video_frame_path)
    
    # Generate labels
    labels = []
    for path in all_image_paths:
        basename = os.path.basename(path)
        if "frame" in basename:
            labels.append("Video Frame")
        elif "salmon_poke_bowl" in basename:
            labels.append("Reference Bowl")
        else:
            # Clean up sample name
            name = basename.replace(".jpg", "").replace(".png", "").replace("_", " ")
            labels.append(name)
    
    print(f"Processing {len(all_image_paths)} images...")
    
    # Generate embeddings using OpenCV
    embeddings, original_images, feature_data = get_opencv_embeddings(all_image_paths)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings, feature_data)
    
    # Visualize similarity matrix
    matrix_path = os.path.join(RESULTS_DIR, "similarity_matrix_opencv.png")
    visualize_similarity_matrix(similarity_matrix, labels, matrix_path)
    print(f"Similarity matrix saved to {matrix_path}")
    
    # Visualize embeddings if we have enough samples
    if len(embeddings) >= 2:
        tsne_path = os.path.join(RESULTS_DIR, "embeddings_visualization_opencv.png")
        visualize_embeddings_tsne(embeddings, labels, original_images, tsne_path)
        print(f"Embeddings visualization saved to {tsne_path}")
    
    # Find most similar to reference image (index 0)
    similar_results = find_most_similar(0, similarity_matrix, labels)
    print("\nMost similar to Reference Bowl:")
    for i, result in enumerate(similar_results[:3]):
        print(f"{i+1}. {result['label']} (similarity: {result['similarity']:.4f})")
    
    # Create visualization of similar images
    similar_indices = [r["index"] for r in similar_results[:3]]
    similar_images = [original_images[i] for i in similar_indices]
    similar_scores = [similarity_matrix[0][i] for i in similar_indices]
    similar_labels = [labels[i] for i in similar_indices]
    
    query_results_path = os.path.join(RESULTS_DIR, "similar_to_reference_opencv.png")
    visualize_query_results(original_images[0], similar_images, similar_scores, similar_labels, query_results_path)
    print(f"Query results visualization saved to {query_results_path}")
    
    # Save embeddings
    embeddings_path = os.path.join(RESULTS_DIR, "bowl_embeddings_opencv.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'image_paths': all_image_paths,
            'labels': labels,
            'method': args.method
        }, f)
    print(f"Embeddings saved to {embeddings_path}")

if __name__ == "__main__":
    main() 