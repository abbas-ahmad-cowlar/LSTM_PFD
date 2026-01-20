#!/bin/bash
# Database Restore Script
# Reference: Master Roadmap Chapter 4.6.2
#
# Usage:
#   ./restore-db.sh                    # Restore from latest backup
#   ./restore-db.sh backup-2026-01-15  # Restore specific backup
#
# Prerequisites:
#   - kubectl configured with cluster access
#   - AWS CLI configured (if using S3)
#   - Appropriate RBAC permissions

set -euo pipefail

# Configuration
NAMESPACE="${NAMESPACE:-default}"
DB_POD_SELECTOR="${DB_POD_SELECTOR:-app=postgresql}"
DB_NAME="${DB_NAME:-lstm_pfd}"
DB_USER="${DB_USER:-postgres}"
BACKUP_BUCKET="${BACKUP_BUCKET:-lstm-pfd-backups}"
BACKUP_PREFIX="${BACKUP_PREFIX:-db}"
LOCAL_BACKUP_DIR="${LOCAL_BACKUP_DIR:-/tmp/lstm-pfd-restore}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    log_info "Prerequisites check passed."
}

# Get database pod
get_db_pod() {
    local pod=$(kubectl get pod -n "$NAMESPACE" -l "$DB_POD_SELECTOR" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "$pod" ]; then
        log_error "Database pod not found with selector: $DB_POD_SELECTOR"
        exit 1
    fi
    
    echo "$pod"
}

# Download backup from S3
download_backup() {
    local backup_name="${1:-latest}"
    
    mkdir -p "$LOCAL_BACKUP_DIR"
    
    if [ "$backup_name" == "latest" ]; then
        log_info "Finding latest backup..."
        backup_name=$(aws s3 ls "s3://$BACKUP_BUCKET/$BACKUP_PREFIX/" | sort | tail -n 1 | awk '{print $4}')
        
        if [ -z "$backup_name" ]; then
            log_error "No backups found in s3://$BACKUP_BUCKET/$BACKUP_PREFIX/"
            exit 1
        fi
    fi
    
    local backup_file="$LOCAL_BACKUP_DIR/$backup_name"
    
    log_info "Downloading backup: $backup_name"
    aws s3 cp "s3://$BACKUP_BUCKET/$BACKUP_PREFIX/$backup_name" "$backup_file"
    
    # Decompress if needed
    if [[ "$backup_name" == *.gz ]]; then
        log_info "Decompressing backup..."
        gunzip -f "$backup_file"
        backup_file="${backup_file%.gz}"
    fi
    
    echo "$backup_file"
}

# Stop application pods
stop_application() {
    log_info "Scaling down application pods..."
    
    kubectl scale deployment -n "$NAMESPACE" lstm-pfd-dashboard --replicas=0 || true
    kubectl scale deployment -n "$NAMESPACE" lstm-pfd-worker --replicas=0 || true
    
    log_info "Waiting for pods to terminate..."
    sleep 10
}

# Restore database
restore_database() {
    local backup_file="$1"
    local db_pod=$(get_db_pod)
    
    log_info "Restoring database from: $backup_file"
    log_info "Target pod: $db_pod"
    
    # Copy backup to pod
    log_info "Copying backup to pod..."
    kubectl cp "$backup_file" "$NAMESPACE/$db_pod:/tmp/restore.sql"
    
    # Drop existing connections
    log_info "Terminating existing connections..."
    kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U "$DB_USER" -c \
        "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '$DB_NAME' AND pid <> pg_backend_pid();" || true
    
    # Drop and recreate database
    log_info "Dropping and recreating database..."
    kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U "$DB_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;"
    kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U "$DB_USER" -c "CREATE DATABASE $DB_NAME;"
    
    # Restore
    log_info "Restoring database..."
    kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U "$DB_USER" -d "$DB_NAME" -f /tmp/restore.sql
    
    # Cleanup
    kubectl exec -n "$NAMESPACE" "$db_pod" -- rm /tmp/restore.sql
    
    log_info "Database restored successfully."
}

# Start application pods
start_application() {
    log_info "Scaling up application pods..."
    
    kubectl scale deployment -n "$NAMESPACE" lstm-pfd-dashboard --replicas=3 || true
    kubectl scale deployment -n "$NAMESPACE" lstm-pfd-worker --replicas=2 || true
    
    log_info "Waiting for pods to be ready..."
    kubectl rollout status deployment/lstm-pfd-dashboard -n "$NAMESPACE" --timeout=300s || true
}

# Verify restoration
verify_restoration() {
    log_info "Verifying restoration..."
    
    local db_pod=$(get_db_pod)
    
    # Check table count
    local table_count=$(kubectl exec -n "$NAMESPACE" "$db_pod" -- psql -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    
    log_info "Table count: $table_count"
    
    # Health check
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "http://lstm-pfd.${NAMESPACE}.svc.cluster.local/health" > /dev/null 2>&1; then
            log_info "Health check passed!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 5
    done
    
    log_warn "Health check did not pass within timeout. Please verify manually."
    return 1
}

# Main execution
main() {
    local backup_name="${1:-latest}"
    local start_time=$(date +%s)
    
    log_info "========================================="
    log_info "LSTM-PFD Database Restoration"
    log_info "========================================="
    log_info "Backup: $backup_name"
    log_info "Namespace: $NAMESPACE"
    log_info "Database: $DB_NAME"
    log_info "========================================="
    
    # Confirm before proceeding
    if [ "${SKIP_CONFIRM:-false}" != "true" ]; then
        read -p "This will replace the current database. Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Restoration cancelled."
            exit 0
        fi
    fi
    
    check_prerequisites
    
    log_info "Step 1/5: Downloading backup..."
    local backup_file=$(download_backup "$backup_name")
    
    log_info "Step 2/5: Stopping application..."
    stop_application
    
    log_info "Step 3/5: Restoring database..."
    restore_database "$backup_file"
    
    log_info "Step 4/5: Starting application..."
    start_application
    
    log_info "Step 5/5: Verifying restoration..."
    verify_restoration || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "========================================="
    log_info "Restoration complete!"
    log_info "Duration: ${duration}s"
    log_info "========================================="
    
    # Cleanup
    rm -f "$backup_file"
}

main "$@"
