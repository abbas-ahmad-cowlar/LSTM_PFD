#!/bin/bash
# Model Backup Script
# Reference: Master Roadmap Chapter 4.6.1
#
# Backs up model checkpoints and training artifacts to S3
#
# Usage:
#   ./backup-models.sh                    # Backup all models
#   ./backup-models.sh --prefix myexp     # Backup with custom prefix
#
# Schedule via cron:
#   0 2 * * * /path/to/backup-models.sh >> /var/log/model-backup.log 2>&1

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
BACKUP_BUCKET="${BACKUP_BUCKET:-lstm-pfd-backups}"
MODEL_PREFIX="${MODEL_PREFIX:-models}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/app/checkpoints}"
OUTPUT_PATH="${OUTPUT_PATH:-/app/outputs}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# Parse arguments
PREFIX=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate backup name
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="${PREFIX:+${PREFIX}-}models-${TIMESTAMP}"

# Check AWS CLI
check_aws() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
}

# Backup local models (if running locally)
backup_local_models() {
    log_info "Backing up local models..."
    
    local local_checkpoint_dir="./checkpoints"
    local local_output_dir="./outputs"
    
    if [ -d "$local_checkpoint_dir" ]; then
        log_info "Uploading checkpoints..."
        aws s3 sync "$local_checkpoint_dir" "s3://$BACKUP_BUCKET/$MODEL_PREFIX/$BACKUP_NAME/checkpoints/" \
            --exclude "*.tmp" \
            --exclude "__pycache__/*"
    fi
    
    if [ -d "$local_output_dir" ]; then
        log_info "Uploading outputs..."
        aws s3 sync "$local_output_dir" "s3://$BACKUP_BUCKET/$MODEL_PREFIX/$BACKUP_NAME/outputs/" \
            --exclude "*.tmp" \
            --exclude "__pycache__/*"
    fi
}

# Backup Kubernetes PVC models
backup_k8s_models() {
    log_info "Backing up Kubernetes models..."
    
    # Create temporary pod for backup
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-backup-job
  namespace: $NAMESPACE
spec:
  containers:
  - name: backup
    image: amazon/aws-cli:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: checkpoints
      mountPath: /checkpoints
  volumes:
  - name: checkpoints
    persistentVolumeClaim:
      claimName: lstm-pfd-checkpoints
  restartPolicy: Never
EOF
    
    # Wait for pod
    kubectl wait --for=condition=Ready pod/model-backup-job -n "$NAMESPACE" --timeout=120s
    
    # Execute backup
    kubectl exec -n "$NAMESPACE" model-backup-job -- \
        aws s3 sync /checkpoints "s3://$BACKUP_BUCKET/$MODEL_PREFIX/$BACKUP_NAME/checkpoints/"
    
    # Cleanup
    kubectl delete pod model-backup-job -n "$NAMESPACE"
}

# Create backup manifest
create_manifest() {
    log_info "Creating backup manifest..."
    
    local manifest_file="/tmp/backup-manifest.json"
    
    cat > "$manifest_file" <<EOF
{
    "backup_name": "$BACKUP_NAME",
    "timestamp": "$(date -Iseconds)",
    "source": {
        "namespace": "$NAMESPACE",
        "checkpoint_path": "$CHECKPOINT_PATH",
        "output_path": "$OUTPUT_PATH"
    },
    "destination": {
        "bucket": "$BACKUP_BUCKET",
        "prefix": "$MODEL_PREFIX/$BACKUP_NAME"
    },
    "retention_days": $RETENTION_DAYS
}
EOF
    
    aws s3 cp "$manifest_file" "s3://$BACKUP_BUCKET/$MODEL_PREFIX/$BACKUP_NAME/manifest.json"
    rm "$manifest_file"
}

# Cleanup old backups
cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days..."
    
    local cutoff_date=$(date -d "-$RETENTION_DAYS days" +%Y%m%d)
    
    # List all backup prefixes
    aws s3 ls "s3://$BACKUP_BUCKET/$MODEL_PREFIX/" | while read -r line; do
        local backup_dir=$(echo "$line" | awk '{print $2}')
        # Extract date from backup name (format: models-YYYYMMDD-HHMMSS/)
        local backup_date=$(echo "$backup_dir" | grep -oP '\d{8}' | head -1)
        
        if [ -n "$backup_date" ] && [ "$backup_date" -lt "$cutoff_date" ]; then
            log_info "Deleting old backup: $backup_dir"
            aws s3 rm "s3://$BACKUP_BUCKET/$MODEL_PREFIX/$backup_dir" --recursive
        fi
    done
}

# Main
main() {
    log_info "========================================="
    log_info "LSTM-PFD Model Backup"
    log_info "========================================="
    log_info "Backup name: $BACKUP_NAME"
    log_info "Destination: s3://$BACKUP_BUCKET/$MODEL_PREFIX/$BACKUP_NAME"
    log_info "========================================="
    
    check_aws
    
    # Try Kubernetes backup first, fall back to local
    if kubectl cluster-info &> /dev/null 2>&1; then
        backup_k8s_models
    else
        backup_local_models
    fi
    
    create_manifest
    cleanup_old_backups
    
    log_info "========================================="
    log_info "Backup complete: $BACKUP_NAME"
    log_info "========================================="
}

main "$@"
