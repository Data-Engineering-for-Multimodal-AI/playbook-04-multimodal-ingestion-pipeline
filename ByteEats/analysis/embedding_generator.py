#!/usr/bin/env python3
"""
ByteEats Video Embedding Generator
---------------------------------
Generates vector embeddings from video features for similarity search and anomaly detection.
Extends the ByteEats pipeline with advanced ML capabilities.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import tensorflow, fall back to dummy embedding if not available
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf_available = True
except ImportError:
    tf_available = False
    print("TensorFlow not available. Using fallback embedding method.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("video-embeddings")

# Configuration
VIDEO_FEATURES_DIR = "video_features_enhanced"
EMBEDDINGS_DIR = "video_embeddings"
MODELS_DIR = "embedding_models"

# Ensure directories exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class VideoEmbeddingGenerator:
    """Generates vector embeddings from video features"""
    
    def __init__(self, features_dir: str, embeddings_dir: str, models_dir: str):
        self.features_dir = features_dir
        self.embeddings_dir = embeddings_dir
        self.models_dir = models_dir
        self.embedding_size = 128  # Default embedding size
        
        # Load or initialize the embedding model
        self.model = self.load_embedding_model()
        
        # Initialize scaler for feature normalization
        self.scaler = StandardScaler()
        
    def load_embedding_model(self) -> Any:
        """Load the embedding model (TensorFlow or fallback)"""
        if tf_available:
            try:
                # Try to load a pretrained model from TF Hub
                logger.info("Loading TensorFlow Hub model for embeddings")
                model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                self.embedding_size = 512  # USE produces 512-dim embeddings
                return model
            except Exception as e:
                logger.warning(f"Error loading TF Hub model: {str(e)}")
                logger.warning("Falling back to simpler embedding method")
                return None
        return None
    
    def extract_features_vector(self, video_data: Dict[str, Any]) -> np.ndarray:
        """Extract a feature vector from video data"""
        features = []
        
        # Extract color features (average of top colors)
        if "color_analyses" in video_data:
            color_features = []
            for color_analysis in video_data["color_analyses"]:
                colors = color_analysis.get("colors", {})
                # Extract top color percentages
                color_values = [
                    colors.get("salmon_red", 0),
                    colors.get("salmon_orange", 0),
                    colors.get("seaweed_green", 0),
                    colors.get("sauce_brown", 0),
                    colors.get("rice_white", 0),
                    colors.get("container_blue", 0),
                    colors.get("yellow_ingredient", 0)
                ]
                color_features.append(color_values)
            
            # Average color features across frames
            if color_features:
                avg_colors = np.mean(color_features, axis=0).tolist()
                features.extend(avg_colors)
        
        # Extract edge features
        if "edge_analyses" in video_data:
            edge_densities = [e.get("edge_density", 0) for e in video_data["edge_analyses"]]
            if edge_densities:
                features.append(np.mean(edge_densities))
                features.append(np.max(edge_densities))
                features.append(np.min(edge_densities))
        
        # Extract motion features
        if "motion_analysis" in video_data:
            motion_data = video_data["motion_analysis"]
            avg_motion = motion_data.get("average_motion", 0)
            features.append(avg_motion)
            
            # Extract motion scores if available
            motion_scores = motion_data.get("motion_scores", [])
            if motion_scores:
                scores = [m.get("motion_score", 0) for m in motion_scores]
                features.append(np.mean(scores))
                features.append(np.max(scores))
                features.append(np.std(scores))
        
        # Add video type and stage as one-hot encoded features
        video_type = video_data.get("video_type", "")
        stage = video_data.get("stage", 0)
        
        # One-hot encode video type
        video_types = ["prep", "assembly", "quality", "packaging"]
        for vt in video_types:
            features.append(1.0 if video_type == vt else 0.0)
        
        # Add stage as a normalized feature
        features.append(stage / 10.0)
        
        # Add detected ingredients as binary features
        ingredients = video_data.get("detected_ingredients", {}).get("ingredients", [])
        all_ingredients = ["salmon", "rice", "seaweed", "sauce", "container"]
        for ingredient in all_ingredients:
            features.append(1.0 if ingredient in ingredients else 0.0)
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def generate_embedding(self, feature_vector: np.ndarray) -> np.ndarray:
        """Generate an embedding from a feature vector"""
        if self.model is not None and tf_available:
            # For text-based models, convert features to string format
            feature_str = " ".join([f"{i}:{v:.4f}" for i, v in enumerate(feature_vector)])
            
            # Generate embedding using TensorFlow model
            embedding = self.model([feature_str]).numpy()[0]
            return embedding
        else:
            # Fallback method: compress features with PCA
            if len(feature_vector) > self.embedding_size:
                # If we have more features than our target embedding size, use PCA
                pca = PCA(n_components=self.embedding_size)
                # Reshape for single sample
                reshaped = feature_vector.reshape(1, -1)
                return pca.fit_transform(reshaped)[0]
            
            # If we have fewer features than embedding size, pad with zeros
            embedding = np.zeros(self.embedding_size)
            embedding[:len(feature_vector)] = feature_vector
            return embedding
    
    def process_video_features(self, order_id: str) -> Dict[str, Any]:
        """Process video features and generate embeddings for an order"""
        # Find combined features file
        combined_file = os.path.join(self.features_dir, f"{order_id}_combined_features.json")
        
        if not os.path.exists(combined_file):
            logger.warning(f"No features file found for order {order_id}")
            return {"success": False, "error": "features_not_found"}
        
        try:
            # Load features
            with open(combined_file, 'r') as f:
                combined_data = json.load(f)
            
            # Get all video features
            all_videos = combined_data.get("features", [])
            
            if not all_videos:
                logger.warning(f"No video features found in {combined_file}")
                return {"success": False, "error": "empty_features"}
            
            # Process each video
            embeddings = []
            feature_vectors = []
            video_paths = []
            
            for video_data in all_videos:
                video_path = video_data.get("video_path", "")
                if not video_path:
                    continue
                
                # Extract feature vector
                feature_vector = self.extract_features_vector(video_data)
                
                # Generate embedding
                embedding = self.generate_embedding(feature_vector)
                
                embeddings.append(embedding)
                feature_vectors.append(feature_vector)
                video_paths.append(os.path.basename(video_path))
            
            # Save embeddings
            embeddings_file = os.path.join(self.embeddings_dir, f"{order_id}_embeddings.pkl")
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    "order_id": order_id,
                    "menu_item": combined_data.get("menu_item", ""),
                    "embeddings": embeddings,
                    "feature_vectors": feature_vectors,
                    "video_paths": video_paths,
                    "timestamp": datetime.now().isoformat()
                }, f)
            
            # Create embedding visualization
            self.visualize_embeddings(embeddings, video_paths, order_id)
            
            logger.info(f"Generated {len(embeddings)} embeddings for order {order_id}")
            
            return {
                "success": True,
                "order_id": order_id,
                "embeddings_count": len(embeddings),
                "embeddings_file": embeddings_file
            }
            
        except Exception as e:
            logger.error(f"Error generating embeddings for {order_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def visualize_embeddings(self, embeddings: List[np.ndarray], labels: List[str], order_id: str) -> None:
        """Visualize embeddings using t-SNE"""
        if not embeddings or len(embeddings) < 2:
            return
        
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)
            
            # Use t-SNE to reduce dimensionality to 2D
            # t-SNE needs perplexity < number of samples (7 videos per order here)
            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=min(30.0, len(embeddings_array) - 1))
            embeddings_2d = tsne.fit_transform(embeddings_array)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', s=100)
            
            # Add labels
            for i, label in enumerate(labels):
                plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                            fontsize=9, ha='center')
            
            plt.title(f"Video Embeddings for Order {order_id}")
            plt.tight_layout()
            
            # Save visualization
            viz_path = os.path.join(self.embeddings_dir, f"{order_id}_embeddings_viz.png")
            plt.savefig(viz_path)
            plt.close()
            
            logger.info(f"Embeddings visualization saved to {viz_path}")
            
        except Exception as e:
            logger.warning(f"Could not create embeddings visualization: {str(e)}")
    
    def find_similar_videos(self, query_video_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find videos similar to a query video using vector similarity"""
        query_order_id = None
        query_video_name = os.path.basename(query_video_path)
        query_embedding = None
        
        # Search through all embedding files to find the query video
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_embeddings.pkl"):
                embedding_path = os.path.join(self.embeddings_dir, file)
                with open(embedding_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                
                # Check if query video is in this file
                video_paths = embedding_data.get("video_paths", [])
                if query_video_name in video_paths:
                    query_order_id = embedding_data.get("order_id")
                    query_idx = video_paths.index(query_video_name)
                    query_embedding = embedding_data.get("embeddings")[query_idx]
                    break
        
        if query_embedding is None:
            logger.warning(f"No embedding found for video {query_video_name}")
            return []
        
        # Compare with all other videos
        all_results = []
        
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_embeddings.pkl"):
                embedding_path = os.path.join(self.embeddings_dir, file)
                with open(embedding_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                
                order_id = embedding_data.get("order_id")
                menu_item = embedding_data.get("menu_item", "")
                video_paths = embedding_data.get("video_paths", [])
                embeddings = embedding_data.get("embeddings", [])
                
                # Calculate similarity for each video
                for i, (video_path, embedding) in enumerate(zip(video_paths, embeddings)):
                    # Skip same video
                    if video_path == query_video_name and order_id == query_order_id:
                        continue
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                    
                    all_results.append({
                        "order_id": order_id,
                        "menu_item": menu_item,
                        "video_path": video_path,
                        "similarity": float(similarity)
                    })
        
        # Sort by similarity (descending)
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k results
        return all_results[:top_k]
    
    def detect_anomalies(self, order_id: str, threshold: float = 0.8) -> Dict[str, Any]:
        """Detect anomalies in video embeddings compared to previous orders"""
        # Load embeddings for current order
        current_embeddings_file = os.path.join(self.embeddings_dir, f"{order_id}_embeddings.pkl")
        
        if not os.path.exists(current_embeddings_file):
            logger.warning(f"No embeddings found for order {order_id}")
            return {"success": False, "error": "embeddings_not_found"}
        
        with open(current_embeddings_file, 'rb') as f:
            current_data = pickle.load(f)
        
        current_videos = current_data.get("video_paths", [])
        current_embeddings = current_data.get("embeddings", [])
        
        if not current_videos or not current_embeddings:
            return {"success": False, "error": "empty_embeddings"}
        
        # Collect embeddings from other orders for comparison
        other_embeddings = {}
        
        for file in os.listdir(self.embeddings_dir):
            if file.endswith("_embeddings.pkl") and not file.startswith(order_id):
                embedding_path = os.path.join(self.embeddings_dir, file)
                with open(embedding_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                
                video_paths = embedding_data.get("video_paths", [])
                embeddings = embedding_data.get("embeddings", [])
                
                for video_path, embedding in zip(video_paths, embeddings):
                    video_type = self.get_video_type_from_path(video_path)
                    if video_type not in other_embeddings:
                        other_embeddings[video_type] = []
                    other_embeddings[video_type].append(embedding)
        
        # Check each video in current order for anomalies
        anomalies = []
        
        for i, (video_path, embedding) in enumerate(zip(current_videos, current_embeddings)):
            video_type = self.get_video_type_from_path(video_path)
            
            # Skip if we don't have comparison data for this video type
            if video_type not in other_embeddings or not other_embeddings[video_type]:
                continue
            
            # Calculate average similarity to other videos of same type
            similarities = []
            for other_emb in other_embeddings[video_type]:
                sim = np.dot(embedding, other_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(other_emb))
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities)
            
            # If similarity is below threshold, mark as anomaly
            if avg_similarity < threshold:
                anomalies.append({
                    "video_path": video_path,
                    "video_type": video_type,
                    "similarity_score": float(avg_similarity),
                    "threshold": threshold
                })
        
        return {
            "success": True,
            "order_id": order_id,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "is_anomalous": len(anomalies) > 0
        }
    
    def get_video_type_from_path(self, video_path: str) -> str:
        """Extract video type from path"""
        video_name = os.path.basename(video_path)
        if "prep" in video_name:
            return "prep"
        elif "assembly" in video_name:
            return "assembly"
        elif "quality" in video_name:
            return "quality"
        elif "packaging" in video_name:
            return "packaging"
        else:
            return "unknown"

def generate_embeddings(order_id: str) -> Dict[str, Any]:
    """Generate embeddings for a specific order"""
    generator = VideoEmbeddingGenerator(VIDEO_FEATURES_DIR, EMBEDDINGS_DIR, MODELS_DIR)
    return generator.process_video_features(order_id)

def find_similar_videos(query_video: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Find videos similar to the query video"""
    generator = VideoEmbeddingGenerator(VIDEO_FEATURES_DIR, EMBEDDINGS_DIR, MODELS_DIR)
    return generator.find_similar_videos(query_video, top_k)

def detect_anomalies(order_id: str, threshold: float = 0.8) -> Dict[str, Any]:
    """Detect anomalies in video embeddings"""
    generator = VideoEmbeddingGenerator(VIDEO_FEATURES_DIR, EMBEDDINGS_DIR, MODELS_DIR)
    return generator.detect_anomalies(order_id, threshold)

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Embedding Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate embeddings command
    generate_parser = subparsers.add_parser("generate", help="Generate embeddings for an order")
    generate_parser.add_argument("order_id", help="Order ID to process")
    
    # Find similar videos command
    similar_parser = subparsers.add_parser("similar", help="Find similar videos")
    similar_parser.add_argument("video_path", help="Path to query video")
    similar_parser.add_argument("--top", type=int, default=3, help="Number of results to return")
    
    # Detect anomalies command
    anomaly_parser = subparsers.add_parser("anomalies", help="Detect anomalies in videos")
    anomaly_parser.add_argument("order_id", help="Order ID to check")
    anomaly_parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        result = generate_embeddings(args.order_id)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)
    elif args.command == "similar":
        results = find_similar_videos(args.video_path, args.top)
        print(json.dumps(results, indent=2))
    elif args.command == "anomalies":
        result = detect_anomalies(args.order_id, args.threshold)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)
    else:
        parser.print_help() 