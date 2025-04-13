# ByteEats Core Ops Pipeline

## Overview

ByteEats is an end-to-end ELT (Extract, Load, Transform) pipeline for automated food preparation. The system processes videos of food preparation, extracts meaningful features, generates vector embeddings for similarity search and anomaly detection, and provides quality assurance through image comparison.

### Pipeline Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Order     │    │   Robot     │    │   Video     │    │   Quality   │
│  Ingestion  │─→  │ Preparation │─→  │  Feature    │─→  │  Assurance  │
│             │    │             │    │ Extraction  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               ▼
                                      ┌─────────────┐    ┌─────────────┐
                                      │ Similarity  │    │   Vector    │
                                      │   Search    │←─  │ Embeddings  │
                                      │             │    │             │
                                      └─────────────┘    └─────────────┘
```

## Pipeline Components

The ByteEats pipeline consists of six main stages:

1. **Order Ingestion** - Receives and processes customer orders
2. **Robot Preparation** - Triggers the robot food preparation sequence
3. **Video Feature Extraction** - Analyzes videos of the preparation process
4. **Quality Assurance** - Compares prepared food against reference images
5. **Vector Embeddings** - Transforms features into mathematical vectors
6. **Similarity Search** - Finds similar food preparations using vector embeddings

## Project Structure

```
ByteEats/
├── api/                   # API endpoints
│   └── order_api.py       # Order ingestion API
├── services/              # Core services
│   ├── robot_trigger.py   # Robot preparation service
│   └── notification.py    # Customer notification service
├── extractors/            # Feature extraction
│   ├── video_extractor.py # Basic video feature extractor
│   └── enhanced_extractor.py # Advanced feature extraction
├── analysis/              # Analysis components
│   ├── quality_checker.py # Image quality comparison
│   ├── embedding_generator.py # Vector embedding generation
│   └── similarity_demo.py # Similarity search demonstration
├── models/                # Data models and schemas
├── data/                  # Sample data
│   ├── reference_images/  # Reference food images
│   ├── sample_bowls/      # Sample food bowl images
│   └── sample-videos/     # Sample preparation videos
├── utils/                 # Utility functions
├── docs/                  # Documentation
│   └── images/            # Documentation images
├── tests/                 # Test cases
├── requirements.txt       # Project dependencies
├── setup.py               # Installation script
├── LICENSE                # License information
└── README.md              # Project documentation
```

## Installation

### Prerequisites

- **Python 3.9+**: Core programming language used for all components
- **OpenCV 4.8+**: Computer vision library for video and image processing
- **NumPy**: Numerical computing library for array operations
- **scikit-learn**: Machine learning library for vector embeddings
- **FastAPI**: API framework for order ingestion
- **Matplotlib**: Data visualization for analysis reports
- **Pillow**: Image processing library for quality checking
- **Pandas**: Data manipulation for feature extraction and analysis

All dependencies are listed in `requirements.txt`.

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ByteEats.git
cd ByteEats
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p orders processed_orders robot_logs video_features_enhanced notification_logs qa_results video_embeddings
```

## Sample Data

The repository includes sample data for demonstration purposes:

### Sample Videos

The `data/sample-videos` directory contains 7 video files that represent a complete food preparation sequence:

1. `salmon-prep-1.mp4` - Initial salmon preparation (3.0MB)
2. `salmon-prep-2.mp4` - Salmon cutting process (3.4MB)
3. `salmon-prep-3.mp4` - Salmon portioning (2.9MB)
4. `salmon-prep-4.mp4` - Completing salmon preparation (2.7MB)
5. `salmon-bowl-assembly-5.mp4` - Bowl assembly with rice and salmon (2.8MB)
6. `quality-check-visual-6.mp4` - Quality verification process (2.9MB)
7. `delivery-packaging-7.mp4` - Final packaging for delivery (2.2MB)

### Reference Images

The `data/reference_images` directory contains high-quality reference images of properly prepared food items:

- `salmon_poke_bowl.jpg` - Reference image of a perfect salmon poke bowl

### Sample Bowl Images

The `data/sample_bowls` directory contains additional images of salmon bowls with different variations:

- `assembly_final.jpg` - Final frame from assembly video
- `final_packaging.jpg` - Final frame from packaging video
- `miso_salmon_bowl.jpg` - Variation with miso sauce
- `salmon_edamame_bowl.jpg` - Variation with edamame
- `salmon_poke_bowl.jpg` - Standard salmon poke bowl

## Running the Pipeline

### 1. Start the Order API

```bash
python ByteEats/api/order_api.py
```

The API will start on http://localhost:8000.

### 2. Process a Sample Order

```bash
curl -X POST "http://localhost:8000/api/process-sample-order"
```

### 3. Run the Robot Trigger Service

```bash
python ByteEats/services/robot_trigger.py
```

### 4. Extract Video Features

```bash
python ByteEats/extractors/enhanced_extractor.py
```

### 5. Run Quality Checks

```bash
python ByteEats/analysis/quality_checker.py ORD1001 "Salmon Poke Bowl"
```

### 6. Generate Vector Embeddings

```bash
python ByteEats/analysis/embedding_generator.py generate ORD1001
```

### 7. Run Similarity Search Demo

```bash
python ByteEats/analysis/similarity_demo.py
```

### Running the Complete Pipeline

For convenience, you can run the entire pipeline with a single command:

```bash
python ByteEats/run_pipeline.py
```

This script will execute all steps in sequence and provide detailed output for each stage.

## Data Engineering Highlights

The ByteEats pipeline showcases several advanced data engineering concepts:

- **Extract-Load-Transform (ELT)**: Converting raw video data into structured features and vector embeddings
- **Computer Vision Integration**: Using OpenCV for extracting meaningful features from video frames
- **Vector Embeddings**: Transforming complex visual data into high-dimensional vectors
- **Similarity Search**: Finding similar items based on vector similarity
- **Anomaly Detection**: Identifying unusual patterns in preparation sequences
- **Service-Oriented Architecture**: Building a pipeline of decoupled services

## Production Considerations

For a production deployment, consider:

1. **Scalability**: 
   - Distributed video processing with Apache Spark
   - Vector databases like FAISS or Pinecone for efficient similarity search

2. **Reliability**:
   - Message queues (Kafka, RabbitMQ) for reliable service communication
   - Service orchestration with Kubernetes

3. **Monitoring**:
   - Metrics collection for pipeline performance
   - Anomaly detection to identify quality issues

4. **Storage Optimization**:
   - Video compression and selective storage
   - Time-to-live policies for raw data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- scikit-learn for machine learning utilities
- FastAPI for the API framework 