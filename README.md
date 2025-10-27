# 🌾 Crop Recommendation System 

An integrated **Machine Learning** framework designed to analyze soil nutrient data in real-time and recommend the most suitable crops for cultivation.  
The system combines **AI-powered recommendation engine** accessible through a mobile app.

---

## 🚀 Project Overview

Modern agriculture requires intelligent systems to optimize soil management and crop selection.  
The **Crop Recommender** project bridges the gap between crop information and data-driven decision-making by collecting environmental parameters and processing them through a predictive model to recommend optimal crops.

**Core objectives:**
- Collect data on soil nutrients (N, P, K), moisture, humidity, and temperature.
- Performe extensive EDA to understand and find hiddent patterns from data.
- Evaluate **Machine Learning models** to recommend the best crop.
- Visualize and manage data through a **Flask-based web application**.

---

## 🧠 System Architecture

```mermaid
graph TD
A[JXBS-3001 Sensor (NPK)] 
C[FC-28 Moisture Sensor] --> B
D[DHT11 Temperature & Humidity] --> B
B -->|MQTT Protocol| E[Cloud Server / Broker]
E --> F[ML Model (Python / Scikit-learn)]
F --> G[Database (Firebase / SQL)]
G --> H[Mobile App (Flutter)]
