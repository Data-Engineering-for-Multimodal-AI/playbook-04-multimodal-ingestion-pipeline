# ByteEats Pipeline Architecture

## Overview

The ByteEats pipeline is an end-to-end ELT (Extract, Load, Transform) system for automated food preparation. The system processes videos of food preparation, extracts meaningful features, generates vector embeddings for similarity search and anomaly detection, and provides quality assurance through image comparison.

## Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Order API   │ -> │ Robot       │ -> │ Video       │ -> │ Pickup      │
│  Ingestion   │    │ Trigger     │    │ Feature     │    │ Notification│
└─────────────┘    └─────────────┘    │ Extraction  │    └─────────────┘
                                      └─────────────┘          ^
                                            │                  │
                                            v                  │
                                      ┌─────────────┐    ┌─────────────┐
                                      │ Vector      │    │ Quality     │
                                      │ Embeddings  │ -> │ Assurance   │
                                      └─────────────┘    └─────────────┘
```

## Components

### 1. Order API (Ingestion)

- **Purpose**: Receives and processes customer orders
- **Input**: Customer order data (menu item, customizations, etc.)
- **Output**: Order JSON files in the `orders` directory
- **Key Files**: `api/order_api.py`

### 2. Robot Trigger

- **Purpose**: Initiates the robot food preparation sequence
- **Input**: Order data from the `orders` directory
- **Output**: Robot commands and logs in `robot_logs` directory
- **Key Files**: `services/robot_trigger.py`

### 3. Video Feature Extraction

- **Purpose**: Analyzes videos of the preparation process
- **Input**: Video files of the food preparation
- **Output**: JSON feature files in `video_features_enhanced` directory
- **Key Files**: `extractors/enhanced_extractor.py`

### 4. Quality Assurance

- **Purpose**: Compares prepared food against reference images
- **Input**: Final video frames and reference images
- **Output**: Quality reports and visualizations in `qa_results` directory
- **Key Files**: `analysis/quality_checker.py`

### 5. Vector Embeddings

- **Purpose**: Transforms features into mathematical vectors
- **Input**: Extracted video features
- **Output**: Vector embeddings in `video_embeddings` directory
- **Key Files**: `analysis/embedding_generator.py`

### 6. Similarity Search

- **Purpose**: Finds similar food preparations using vector embeddings
- **Input**: Vector embeddings
- **Output**: Similarity visualizations in `similarity_results` directory
- **Key Files**: `analysis/similarity_demo.py`

### 7. Pickup Notification

- **Purpose**: Notifies customers when their order is ready
- **Input**: Processed order data and quality checks
- **Output**: Notification logs in `notification_logs` directory
- **Key Files**: `services/notification.py`

## Data Flow

1. Customer places an order through the API
2. Order data is saved as a JSON file
3. Robot trigger service detects the new order and initiates preparation
4. Preparation videos are analyzed for features
5. Features are used for quality assurance and converted to vector embeddings
6. Vector embeddings enable similarity search and anomaly detection
7. When preparation is complete, the customer is notified

## Running the Pipeline

See the main README.md file for instructions on running the complete pipeline. 