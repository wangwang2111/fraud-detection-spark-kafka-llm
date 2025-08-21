# 📞 Real-Time Phone Scam Detection System

This project provides a scalable, real-time scam detection system for transcribed phone dialogues using interpretable machine learning (Decision Trees), Kafka streaming, and LLM-based explanation. It includes a Spark-based backend for model training and a Streamlit frontend with DeepSeek-powered analysis.

## ✅ Requirements

- Python ≥ 3.10
- Kafka (kafka_2.13-2.8.1)
- Install dependencies with:

```bash
pip install -r requirements.txt
```

### Required Libraries:
```bash
pyspark==3.5.5  
pandas==2.2.3  
numpy==1.26.3  
matplotlib==3.9.0  
seaborn==0.13.2  
confluent_kafka==2.9.0  
streamlit==1.44.1  
python-dotenv  
requests  
tenacity==8.3.0
```

## 📂 Project Structure

```
├── fraud_detection_spark.py          # Model training, evaluation, selection, and saving
├── utils/
│   ├── kafka_utils.py               # Kafka integration utilities
│   ├── agent_api.py                 # DeepSeek API + prediction wrapper
│   └── .env                           # Configuration file (API keys, paths, etc.)
├── app_ui.py                        # Streamlit interface for real-time prediction and explanation
├── dialogue_classification_model/   # Saved Spark pipeline model
├── requirements.txt                 # All versions of required libraries
```

## 🚀 How to Run the Project

### 1. Train, Evaluate, and Save the Classification Model

Run the backend Spark pipeline to process, train, and save the model:

```bash
python fraud_detection_spark.py
```

This script will:
- Load and preprocess the dataset
- Train Decision Tree, Random Forest, and XGBoost classifiers comparison
- Evaluate and compare metrics
- Save the best model (default: `dialogue_classification_model/`)

### 2. Set Up Kafka for Real-Time Inference (via Docker)

To enable real-time classification, this project uses **Kafka and Zookeeper containers** via Docker.

#### 🐳 Step 1: Start Kafka & Zookeeper with Docker

Create a file called `docker-compose.yml` (if you haven’t already) and add the following:

```yaml
version: 'latest'

services:
  zookeeper:
    image: wurstmeister/zookeeper
    container_name: zookeeper
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka
    container_name: kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_HOST_NAME: localhost
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
```

Then run:

```bash
docker-compose -f docker-compose.yml up -d
```

#### 🧵 Step 2: Create Required Kafka Topics

After the containers are up, create the topics needed for the app:

```bash
docker exec -it <kafka_conatiner_id> /bin/sh
```

Go inside kafka installation folder
```bash
cd /opt/kafka_<version>/bin
```

Once inside the container:

```bash
# Create input topic
kafka-topics.sh --create \
  --topic customer-dialogues-raw \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Create output topic
kafka-topics.sh --create \
  --topic dialogues-classified \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```


### 3. 🔐 Configure Environment Variables

Edit `.env` or Create a `.env` file in the root directory:

```
# DeepSeek API Key (obtain from https://platform.deepseek.com)
DEEPSEEK_API_KEY=your_deepseek_api_key

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka-server:9092
KAFKA_INPUT_TOPIC=customer-dialogues-raw
KAFKA_OUTPUT_TOPIC=dialogues-classified
KAFKA_CONSUMER_GROUP=dialogue-classifier-group

# Optional: If using secure Kafka (SASL_SSL)
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_USERNAME=your_username
KAFKA_PASSWORD=your_password
```

> 💡 *The `.env` file is loaded automatically by `agent_api.py`.*

### 4. Launch the Real-Time Frontend

Start the Streamlit app:

```bash
streamlit run app_ui.py
```
Features:
- Input a single dialogue or batch of transcriptions
- Get scam classification and confidence score
- View an AI-generated explanation using DeepSeek

## 🧠 DeepSeek Integration

- DeepSeek API is used for generating human-readable justifications.
- Output includes:
  - Key red flags (e.g., “verify your Social Security number”)
  - Summary analysis
  - Suggested next actions

## 📬 Contact

For questions, suggestions, or collaboration, feel free to reach out. 
