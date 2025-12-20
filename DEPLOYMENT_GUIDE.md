# Deployment Guide - Production-Ready AI Platform

## 🎯 Overview

This guide provides comprehensive instructions for deploying the AI-powered image classification platform in production environments, from development to enterprise-scale deployments.

## 📋 Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores (Intel i5 or equivalent)
- **RAM**: 8 GB
- **Storage**: 20 GB free space
- **OS**: Ubuntu 20.04+, macOS 11+, Windows 10+ with WSL2
- **Python**: 3.8 - 3.11
- **Internet**: Stable connection for package downloads

#### Recommended Requirements (Production)
- **CPU**: 8+ cores (Intel Xeon or AMD EPYC)
- **RAM**: 32 GB+
- **GPU**: NVIDIA T4, V100, or A100 (8-16 GB VRAM)
- **Storage**: 100 GB+ SSD (NVMe preferred)
- **OS**: Ubuntu 20.04 LTS or 22.04 LTS
- **Network**: 1 Gbps+

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU support)
- Git 2.30+
- CUDA 11.8+ (for GPU training)
- Node.js 16+ (for monitoring dashboards)

## 🚀 Deployment Options

### Option 1: Local Development

#### 1.1 Clone Repository
```bash
git clone https://github.com/MarceloClaro/CLASSIFICACAO-DE-ROCHAS.git
cd CLASSIFICACAO-DE-ROCHAS
```

#### 1.2 Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Using conda
conda create -n diagnosticai python=3.10
conda activate diagnosticai
```

#### 1.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 1.4 Run Application
```bash
# Streamlit UI
streamlit run app5.py --server.port 8501

# Access at: http://localhost:8501
```

### Option 2: Docker Deployment

#### 2.1 Create Dockerfile
```dockerfile
# Save as Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app5.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2.2 Create Docker Compose
```yaml
# Save as docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./dataset:/app/dataset
      - ./models:/app/models
      - ./results:/app/results
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: PostgreSQL for metadata
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: diagnosticai
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

#### 2.3 Build and Run
```bash
# Build image
docker-compose build

# Run services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down

# Clean up (including volumes)
docker-compose down -v
```

### Option 3: Kubernetes Deployment

#### 3.1 Create Kubernetes Manifests

**Namespace**:
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: diagnosticai
```

**Deployment**:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diagnosticai-app
  namespace: diagnosticai
  labels:
    app: diagnosticai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: diagnosticai
  template:
    metadata:
      labels:
        app: diagnosticai
    spec:
      containers:
      - name: app
        image: diagnosticai:latest
        ports:
        - containerPort: 8501
          name: http
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: STREAMLIT_SERVER_HEADLESS
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: data
          mountPath: /app/dataset
        - name: models
          mountPath: /app/models
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: diagnosticai-data-pvc
      - name: models
        persistentVolumeClaim:
          claimName: diagnosticai-models-pvc
      nodeSelector:
        accelerator: nvidia-gpu
```

**Service**:
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: diagnosticai-service
  namespace: diagnosticai
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
    protocol: TCP
    name: http
  selector:
    app: diagnosticai
```

**Ingress** (with TLS):
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: diagnosticai-ingress
  namespace: diagnosticai
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - diagnosticai.yourdomain.com
    secretName: diagnosticai-tls
  rules:
  - host: diagnosticai.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: diagnosticai-service
            port:
              number: 80
```

**HPA** (Horizontal Pod Autoscaler):
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: diagnosticai-hpa
  namespace: diagnosticai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: diagnosticai-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 3.2 Deploy to Kubernetes
```bash
# Create namespace
kubectl apply -f namespace.yaml

# Create PVCs (define separately based on your storage class)
kubectl apply -f pvc.yaml

# Deploy application
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check status
kubectl get pods -n diagnosticai
kubectl get svc -n diagnosticai
kubectl get ingress -n diagnosticai

# View logs
kubectl logs -f deployment/diagnosticai-app -n diagnosticai

# Scale manually (if needed)
kubectl scale deployment/diagnosticai-app --replicas=5 -n diagnosticai
```

### Option 4: Cloud Platform Deployment

#### 4.1 AWS Deployment

**Using ECS (Elastic Container Service)**:
```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name diagnosticai

# 2. Build and push Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t diagnosticai .
docker tag diagnosticai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/diagnosticai:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/diagnosticai:latest

# 3. Create ECS cluster
aws ecs create-cluster --cluster-name diagnosticai-cluster

# 4. Create task definition (see JSON below)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 5. Create service
aws ecs create-service \
  --cluster diagnosticai-cluster \
  --service-name diagnosticai-service \
  --task-definition diagnosticai:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

**Task Definition** (task-definition.json):
```json
{
  "family": "diagnosticai",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "diagnosticai-app",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/diagnosticai:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STREAMLIT_SERVER_HEADLESS",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/diagnosticai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 4.2 Google Cloud Platform (GCP)

**Using Cloud Run**:
```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/diagnosticai

# 2. Deploy to Cloud Run
gcloud run deploy diagnosticai \
  --image gcr.io/PROJECT_ID/diagnosticai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1

# 3. Get service URL
gcloud run services describe diagnosticai --region us-central1 --format 'value(status.url)'
```

#### 4.3 Azure Deployment

**Using Azure Container Instances**:
```bash
# 1. Create resource group
az group create --name diagnosticai-rg --location eastus

# 2. Create container registry
az acr create --resource-group diagnosticai-rg --name diagnosticaiacr --sku Basic

# 3. Build and push image
az acr build --registry diagnosticaiacr --image diagnosticai:latest .

# 4. Deploy container
az container create \
  --resource-group diagnosticai-rg \
  --name diagnosticai-app \
  --image diagnosticaiacr.azurecr.io/diagnosticai:latest \
  --cpu 4 \
  --memory 8 \
  --registry-login-server diagnosticaiacr.azurecr.io \
  --registry-username <username> \
  --registry-password <password> \
  --dns-name-label diagnosticai \
  --ports 8501
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:
```bash
# Application Settings
APP_NAME=DiagnostiCAI
APP_VERSION=5.0.0
DEBUG=false

# Server Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Model Settings
DEFAULT_MODEL=ResNet50
MODEL_PATH=/app/models
BATCH_SIZE=16
NUM_WORKERS=4

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=/app/.torch

# API Keys (store in secrets manager in production)
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost:5432/diagnosticai
REDIS_URL=redis://localhost:6379/0

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
SENTRY_DSN=your_sentry_dsn

# Security
JWT_SECRET=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
ALLOWED_ORIGINS=https://yourdomain.com

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### Secrets Management

**For Kubernetes** (using secrets):
```bash
# Create secret
kubectl create secret generic diagnosticai-secrets \
  --from-literal=google-api-key=YOUR_KEY \
  --from-literal=groq-api-key=YOUR_KEY \
  --from-literal=jwt-secret=YOUR_SECRET \
  -n diagnosticai

# Use in deployment
env:
  - name: GOOGLE_API_KEY
    valueFrom:
      secretKeyRef:
        name: diagnosticai-secrets
        key: google-api-key
```

**For AWS** (using Secrets Manager):
```bash
# Store secret
aws secretsmanager create-secret \
  --name diagnosticai/api-keys \
  --secret-string '{"google_api_key":"YOUR_KEY","groq_api_key":"YOUR_KEY"}'

# Retrieve in application
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='diagnosticai/api-keys')
secrets = json.loads(response['SecretString'])
```

## 🔒 Security Hardening

### SSL/TLS Configuration

**Using Let's Encrypt** (with Cert-Manager in K8s):
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### Firewall Rules

**Using UFW** (Ubuntu):
```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow Streamlit (only from specific IPs)
sudo ufw allow from YOUR_IP to any port 8501

# Enable firewall
sudo ufw enable
```

### HTTPS Redirect

**NGINX Configuration**:
```nginx
# /etc/nginx/sites-available/diagnosticai
server {
    listen 80;
    server_name diagnosticai.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name diagnosticai.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/diagnosticai.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/diagnosticai.yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 Monitoring & Logging

### Prometheus + Grafana

**docker-compose.monitoring.yml**:
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
```

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'diagnosticai'
    static_configs:
      - targets: ['app:8501']
```

### ELK Stack (Elasticsearch, Logstash, Kibana)

**docker-compose.elk.yml**:
```yaml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

## 🧪 Testing Deployment

### Health Check
```bash
# HTTP request
curl http://localhost:8501/_stcore/health

# Expected response: {"status": "ok"}
```

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test (100 requests, 10 concurrent)
ab -n 100 -c 10 http://localhost:8501/

# Install Locust for more advanced testing
pip install locust

# Create locustfile.py
from locust import HttpUser, task, between

class DiagnostiCAIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def health_check(self):
        self.client.get("/_stcore/health")

# Run Locust
locust -f locustfile.py --host=http://localhost:8501
```

## 🔄 CI/CD Pipeline

### GitHub Actions

**.github/workflows/deploy.yml**:
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t diagnosticai:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push diagnosticai:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        uses: azure/k8s-deploy@v1
        with:
          manifests: |
            k8s/deployment.yaml
            k8s/service.yaml
          images: |
            diagnosticai:${{ github.sha }}
          namespace: diagnosticai
```

## 🆘 Troubleshooting

### Common Issues

**Issue 1: CUDA Out of Memory**
```python
# Solution: Reduce batch size or use gradient checkpointing
# In app5.py, modify:
batch_size = 8  # Reduce from 16
torch.cuda.empty_cache()  # Clear cache between batches
```

**Issue 2: Streamlit "Address already in use"**
```bash
# Find process using port 8501
lsof -i :8501
# or
netstat -tulpn | grep 8501

# Kill process
kill -9 <PID>

# Or use different port
streamlit run app5.py --server.port 8502
```

**Issue 3: Module Import Errors**
```bash
# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt --no-cache-dir
```

**Issue 4: Docker Build Fails**
```bash
# Clear Docker cache
docker builder prune

# Build with no cache
docker build --no-cache -t diagnosticai .
```

## 📚 Additional Resources

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [AWS ECS Guide](https://docs.aws.amazon.com/ecs/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)

## 📞 Support

For deployment support:
- Email: marceloclaro@gmail.com
- WhatsApp: +55 88 98158-7145
- Documentation: Check repository wiki

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Maintained By**: DiagnostiCAI DevOps Team
