# Docker Examples

This document provides examples of how to use rust-jieba with Docker.

## Quick Start

### Build and Run

```bash
# Build the Docker image
docker build -t rust-jieba .

# Run the test script
docker run rust-jieba
```

## Examples

### 1. Basic Usage

```bash
# Run interactive container
docker run -it --rm rust-jieba python -c "
import rust_jieba
text = '我爱北京天安门'
result = rust_jieba.cut(text)
print('/'.join(result))
"
```

### 2. Custom Dictionary

```bash
# Create a custom dictionary
echo "人工智能 1000 n" > custom_dict.txt
echo "机器学习 800 n" >> custom_dict.txt

# Run with custom dictionary
docker run -v $(pwd)/custom_dict.txt:/app/custom_dict.txt rust-jieba python -c "
import rust_jieba
rust_jieba.load_userdict('/app/custom_dict.txt')
text = '人工智能技术在机器学习中的应用'
result = rust_jieba.cut(text)
print('/'.join(result))
"
```

### 3. Batch Processing

```bash
# Create input file
echo -e "我爱北京天安门\n人工智能很棒\n机器学习有趣" > input.txt

# Process file
docker run -v $(pwd)/input.txt:/app/input.txt rust-jieba python -c "
import rust_jieba
with open('/app/input.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            words = rust_jieba.cut(line)
            print(f'{line} => {\"/\".join(words)}')
"
```

### 4. Performance Comparison

```bash
# Run performance test
docker run rust-jieba python -c "
import rust_jieba
import time
import jieba

text = '在这个快速发展的时代，人工智能技术正在改变我们的生活方式。' * 100

# Test Rust jieba
start = time.time()
for _ in range(1000):
    list(rust_jieba.cut(text))
rust_time = time.time() - start

# Test Python jieba
start = time.time()
for _ in range(1000):
    list(jieba.cut(text))
py_time = time.time() - start

print(f'Rust jieba: {rust_time:.4f}s')
print(f'Python jieba: {py_time:.4f}s')
print(f'Speedup: {py_time/rust_time:.2f}x')
"
```

### 5. Web Service Example

Create a simple Flask web service:

```python
# app.py
from flask import Flask, request, jsonify
import rust_jieba

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    mode = data.get('mode', 'default')

    if mode == 'search':
        words = list(rust_jieba.cut_for_search(text))
    elif mode == 'full':
        words = list(rust_jieba.cut(text, cut_all=True))
    else:
        words = list(rust_jieba.cut(text))

    return jsonify({
        'text': text,
        'mode': mode,
        'words': words,
        'segmented': '/'.join(words)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run the web service:

```bash
# Build and run
docker build -t rust-jieba-web .
docker run -p 5000:5000 rust-jieba python -c "
import sys
sys.path.insert(0, '/app')
from app import app
app.run(host='0.0.0.0', port=5000)
"

# Test the API
curl -X POST http://localhost:5000/segment \
  -H "Content-Type: application/json" \
  -d '{"text": "我爱北京天安门", "mode": "default"}'
```

## Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rust-jieba:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./custom_dict.txt:/app/custom_dict.txt:ro
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rust-jieba
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rust-jieba
  template:
    metadata:
      labels:
        app: rust-jieba
    spec:
      containers:
      - name: rust-jieba
        image: rust-jieba:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: rust-jieba-service
spec:
  selector:
    app: rust-jieba
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

## Environment Variables

- `PYTHONUNBUFFERED`: Disable Python output buffering
- `RUST_LOG`: Set Rust log level (debug, info, warn, error)
- `PYTHONPATH`: Set Python path

## Health Checks

The Docker image includes a built-in health check:

```bash
# Check health
docker exec <container_id> python -c "import rust_jieba; print('OK')"
```

## Performance Tips

1. **Reuse tokenizer instances** for better performance
2. **Load custom dictionaries** once at startup
3. **Use batch processing** for multiple texts
4. **Choose appropriate modes** (default, search, full) based on use case

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase memory limits for large texts
2. **Build Failures**: Check Rust toolchain and Python versions
3. **Import Errors**: Ensure the library was built correctly

### Debug Mode

```bash
# Run with debug logging
docker run -e RUST_LOG=debug rust-jieba
```