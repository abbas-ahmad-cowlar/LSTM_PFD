#!/usr/bin/env python3
"""
OpenAPI Specification Generator

Generates OpenAPI (Swagger) spec from the FastAPI application
for embedding in MkDocs documentation.

Usage:
    python scripts/generate_openapi.py

Output:
    docs/api/openapi.json
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_openapi_spec():
    """
    Generate OpenAPI specification from FastAPI app.
    
    Attempts to import the FastAPI application and extract its OpenAPI schema.
    Falls back to a minimal spec if the app cannot be imported.
    """
    output_path = project_root / "docs" / "api" / "openapi.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try to import the FastAPI app
        from packages.deployment.inference_api import app
        
        openapi_schema = app.openapi()
        
        # Add additional metadata
        openapi_schema["info"]["x-logo"] = {
            "url": "https://github.com/abbas-ahmad-cowlar/LSTM_PFD"
        }
        
        with open(output_path, 'w') as f:
            json.dump(openapi_schema, f, indent=2)
        
        print(f"✓ OpenAPI spec generated: {output_path}")
        print(f"  - {len(openapi_schema.get('paths', {}))} endpoints documented")
        return True
        
    except ImportError as e:
        print(f"⚠ Could not import FastAPI app: {e}")
        print("  Generating minimal OpenAPI spec from API_REFERENCE.md...")
        
        # Generate minimal spec based on documented API
        minimal_spec = generate_minimal_spec()
        
        with open(output_path, 'w') as f:
            json.dump(minimal_spec, f, indent=2)
        
        print(f"✓ Minimal OpenAPI spec generated: {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to generate OpenAPI spec: {e}")
        return False


def generate_minimal_spec():
    """
    Generate a minimal OpenAPI spec based on documented endpoints.
    Used as fallback when FastAPI app cannot be imported.
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "LSTM_PFD Fault Diagnosis API",
            "description": "REST API for bearing fault prediction using deep learning models.",
            "version": "1.0.0",
            "contact": {
                "name": "LSTM_PFD Team",
                "url": "https://github.com/abbas-ahmad-cowlar/LSTM_PFD"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ],
        "paths": {
            "/": {
                "get": {
                    "summary": "Root Information",
                    "description": "Get API information and available endpoints.",
                    "operationId": "root_info",
                    "responses": {
                        "200": {
                            "description": "API information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/RootInfo"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check API server health status.",
                    "operationId": "health_check",
                    "responses": {
                        "200": {
                            "description": "Health status",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthStatus"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/model/info": {
                "get": {
                    "summary": "Model Information",
                    "description": "Get information about the loaded model.",
                    "operationId": "model_info",
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelInfo"
                                    }
                                }
                            }
                        },
                        "503": {
                            "description": "Model not loaded"
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Single Prediction",
                    "description": "Predict fault class for a single vibration signal.",
                    "operationId": "predict_single",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PredictionResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid input"
                        }
                    }
                }
            },
            "/predict/batch": {
                "post": {
                    "summary": "Batch Prediction",
                    "description": "Predict fault classes for multiple signals in one request.",
                    "operationId": "predict_batch",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/BatchPredictionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Batch prediction results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BatchPredictionResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "RootInfo": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "example": "LSTM_PFD Fault Diagnosis API"},
                        "version": {"type": "string", "example": "1.0.0"},
                        "docs": {"type": "string", "example": "/docs"},
                        "health": {"type": "string", "example": "/health"}
                    }
                },
                "HealthStatus": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "healthy"},
                        "model_loaded": {"type": "boolean", "example": True},
                        "device": {"type": "string", "example": "cuda"},
                        "version": {"type": "string", "example": "1.0.0"},
                        "uptime_seconds": {"type": "integer", "example": 3600}
                    }
                },
                "ModelInfo": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "example": "EnsembleModel"},
                        "model_version": {"type": "string", "example": "1.0.0"},
                        "num_classes": {"type": "integer", "example": 11},
                        "input_shape": {"type": "array", "items": {"type": "integer"}, "example": [1, 102400]},
                        "inference_device": {"type": "string", "example": "cuda"}
                    }
                },
                "PredictionRequest": {
                    "type": "object",
                    "required": ["signal"],
                    "properties": {
                        "signal": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Vibration signal (102,400 samples at 20,480 Hz)"
                        },
                        "return_probabilities": {"type": "boolean", "default": False},
                        "return_features": {"type": "boolean", "default": False}
                    }
                },
                "PredictionResponse": {
                    "type": "object",
                    "properties": {
                        "predicted_class": {"type": "integer", "example": 2},
                        "class_name": {"type": "string", "example": "Inner Race Fault"},
                        "confidence": {"type": "number", "example": 0.967},
                        "inference_time_ms": {"type": "number", "example": 42.3},
                        "probabilities": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                },
                "BatchPredictionRequest": {
                    "type": "object",
                    "required": ["signals"],
                    "properties": {
                        "signals": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "description": "Array of signals (max 32)"
                        },
                        "return_probabilities": {"type": "boolean", "default": False}
                    }
                },
                "BatchPredictionResponse": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/PredictionResponse"}
                        },
                        "batch_size": {"type": "integer"},
                        "total_inference_time_ms": {"type": "number"},
                        "avg_time_per_sample_ms": {"type": "number"}
                    }
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        },
        "security": [
            {"ApiKeyAuth": []}
        ]
    }


if __name__ == "__main__":
    success = generate_openapi_spec()
    sys.exit(0 if success else 1)
