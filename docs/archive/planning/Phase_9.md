## **PHASE 9: Production Deployment & Edge Optimization**

### Phase Objective
Deploy best models to production environments. Optimize for edge devices (quantization, pruning, ONNX export). Build REST API, containerize with Docker, implement monitoring. Achieve <50ms inference latency on edge hardware with <5% accuracy degradation.

### Complete File List (12 files)

#### **1. Model Optimization (4 files)**

**`deployment/quantization.py`**
- **Purpose**: Quantize models from FP32 to INT8
- **Key Functions**:
  - `quantize_model(model, calibration_data)`:
    ```python
    # PyTorch quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate on representative data
    for batch in calibration_data:
        model(batch)
    
    # Convert to INT8
    torch.quantization.convert(model, inplace=True)
    return model
    ```
  - `benchmark_quantized_model(original, quantized, test_loader)`: Compare accuracy/speed

**`deployment/pruning.py`**
- **Purpose**: Remove redundant weights
- **Key Functions**:
  - `prune_model(model, sparsity=0.5)`:
    ```python
    # Magnitude-based pruning
    for module in model.modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=sparsity)
    ```
  - `fine_tune_pruned_model(pruned_model, train_loader)`: Recover accuracy

**`deployment/onnx_export.py`**
- **Purpose**: Export to ONNX for cross-platform inference
- **Key Functions**:
  - `export_to_onnx(model, save_path, input_shape)`:
    ```python
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=13,
        input_names=['signal'],
        output_names=['class_probabilities'],
        dynamic_axes={'signal': {0: 'batch_size'}}
    )
    ```
  - `validate_onnx_export(original_model, onnx_path, test_samples)`: Check equivalence

**`deployment/knowledge_distillation_deploy.py`**
- **Purpose**: Train small student for edge deployment
- **Target**: ResNet-18 (student) matches ResNet-50 (teacher) with 3× fewer parameters

#### **2. API & Serving (4 files)**

**`deployment/inference_engine.py`**
- **Purpose**: Production inference class
- **Key Classes**:
  - `InferenceEngine`: Load model, preprocess, predict, postprocess
- **Key Functions**:
  - `load_model(model_path)`: Load quantized/ONNX model
  - `predict(signal, return_confidence=True)`:
    ```python
    # Preprocess
    signal_preprocessed = self.preprocess(signal)
    
    # Inference
    with torch.no_grad():
        logits = self.model(signal_preprocessed)
        probabilities = torch.softmax(logits, dim=-1)
    
    # Postprocess
    predicted_class = torch.argmax(probabilities)
    confidence = probabilities.max().item()
    
    return {
        'predicted_fault': self.class_names[predicted_class],
        'confidence': confidence,
        'all_probabilities': probabilities.tolist()
    }
    ```

**`deployment/rest_api.py`**
- **Purpose**: REST API for model serving
- **Endpoints**:
  - `POST /predict`: Upload signal, get prediction
  - `GET /health`: Health check
  - `GET /model_info`: Model metadata (version, accuracy, latency)
- **Technology**: FastAPI
- **Example**:
  ```python
  from fastapi import FastAPI, File, UploadFile
  
  app = FastAPI()
  engine = InferenceEngine(model_path='model.onnx')
  
  @app.post("/predict")
  async def predict(file: UploadFile = File(...)):
      signal = np.load(file.file)
      result = engine.predict(signal)
      return result
  ```

**`deployment/docker_config/`**
- **Files**:
  - `Dockerfile`: Container definition
  - `docker-compose.yml`: Multi-container orchestration
  - `requirements.txt`: Python dependencies
- **Docker Image**: Based on `python:3.9-slim`, includes ONNX Runtime

**`deployment/monitoring.py`**
- **Purpose**: Monitor model performance in production
- **Key Functions**:
  - `log_prediction(signal_id, prediction, confidence, latency)`: Log to database
  - `detect_data_drift(recent_signals, training_distribution)`: Kolmogorov-Smirnov test
  - `trigger_retraining(drift_detected)`: Alert when retraining needed

#### **3. Edge Deployment (2 files)**

**`deployment/edge_inference.py`**
- **Purpose**: Optimized inference for edge devices (Raspberry Pi, Jetson Nano)
- **Optimizations**:
  - INT8 quantization (4× memory reduction)
  - TensorRT optimization (NVIDIA devices)
  - ONNX Runtime with CPU optimizations
- **Target Latency**: <50ms per sample

**`deployment/mobile_deployment.py`**
- **Purpose**: Convert model to TensorFlow Lite for mobile
- **Key Functions**:
  - `convert_to_tflite(model)`: Export to `.tflite`
  - `benchmark_mobile(tflite_model, test_signals)`: Latency on mobile CPU

#### **4. Testing & Validation (2 files)**

**`tests/test_deployment.py`**
- **Unit Tests**: API endpoints, inference engine
- **Integration Tests**: End-to-end prediction pipeline
- **Load Tests**: API handles 100 requests/second

**`deployment/deployment_validator.py`**
- **Purpose**: Validate deployment before production
- **Checks**:
  - Model accuracy matches expected (within 1%)
  - Latency < threshold (50ms)
  - API returns correct schema
  - Docker container starts successfully

### Acceptance Criteria

✅ **Model optimization successful**
- Quantized model: 3-5% accuracy drop, 4× smaller, 3× faster
- Pruned model: 50% sparsity, 2% accuracy drop
- ONNX export validated (predictions match PyTorch)

✅ **API functional**
- REST API handles requests
- Docker container deploys successfully
- Latency < 50ms per prediction

✅ **Edge deployment working**
- Model runs on Raspberry Pi 4 / Jetson Nano
- Inference latency < 50ms
- Accuracy within 5% of server model

✅ **Monitoring implemented**
- Predictions logged to database
- Data drift detection alerts
- Retraining triggers configured

✅ **Documentation complete**
- Deployment guide: "From Model to Production"
- API documentation (Swagger/OpenAPI)
- Docker quickstart guide

### Estimated Effort

**Total: ~14 days (3 weeks) for Phase 9**

**Complexity**: ⭐⭐⭐⭐☆ (High) - DevOps + ML Engineering
