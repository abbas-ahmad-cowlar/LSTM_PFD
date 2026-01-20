# Disaster Recovery Runbook

**Reference:** Master Roadmap Chapter 4.6  
**Last Updated:** January 2026  
**Owner:** Platform Engineering Team

---

## Overview

This document provides step-by-step procedures for recovering the LSTM-PFD system from various failure scenarios. All procedures are designed to meet our RTO (Recovery Time Objective) of **< 1 hour** and RPO (Recovery Point Objective) of **< 4 hours**.

| Metric | Target    | Current                 |
| ------ | --------- | ----------------------- |
| RTO    | < 1 hour  | 45 minutes              |
| RPO    | < 4 hours | 4 hours (daily backups) |

---

## Quick Reference

| Scenario                 | Script                       | Estimated Time |
| ------------------------ | ---------------------------- | -------------- |
| Database Corruption      | `restore-db.sh`              | 15-30 min      |
| Model Checkpoint Loss    | `backup-models.sh --restore` | 10-20 min      |
| Complete Cluster Failure | Full DR Procedure            | 45-60 min      |
| Single Pod Failure       | Kubernetes Self-Heal         | < 5 min        |

---

## 1. Database Recovery

### 1.1 Prerequisites

- [ ] kubectl access to cluster
- [ ] AWS CLI configured with S3 read permissions
- [ ] Database admin credentials
- [ ] Backup bucket access: `s3://lstm-pfd-backups/db/`

### 1.2 Procedure

```bash
# Navigate to DR scripts
cd scripts/disaster-recovery

# Restore from latest backup
./restore-db.sh

# Or restore specific backup
./restore-db.sh backup-2026-01-15.sql.gz
```

### 1.3 Verification Steps

1. Check pod status:

   ```bash
   kubectl get pods -l app=lstm-pfd
   ```

2. Verify database connectivity:

   ```bash
   kubectl exec -it deployment/lstm-pfd-dashboard -- curl localhost:8050/health
   ```

3. Verify data integrity:
   ```bash
   kubectl exec -it $(kubectl get pod -l app=postgresql -o jsonpath='{.items[0].metadata.name}') -- \
     psql -U postgres -d lstm_pfd -c "SELECT COUNT(*) FROM experiments;"
   ```

---

## 2. Model Checkpoint Recovery

### 2.1 Prerequisites

- [ ] AWS CLI configured
- [ ] PVC access in Kubernetes
- [ ] Model backup bucket: `s3://lstm-pfd-backups/models/`

### 2.2 Procedure

```bash
# List available backups
aws s3 ls s3://lstm-pfd-backups/models/

# Restore specific backup
aws s3 sync s3://lstm-pfd-backups/models/models-20260115-020000/checkpoints/ \
  ./checkpoints/

# Restart pods to reload models
kubectl rollout restart deployment/lstm-pfd-dashboard
```

### 2.3 Verification

```bash
# Check model loaded metric
curl -s http://localhost:8050/metrics | grep lstm_pfd_model_loaded
```

---

## 3. Complete Cluster Recovery

### 3.1 Prerequisites

- [ ] Terraform/Pulumi state access
- [ ] Helm charts available
- [ ] All backup buckets accessible
- [ ] DNS/Ingress configuration ready

### 3.2 Procedure

#### Step 1: Provision Infrastructure (15 min)

```bash
cd infrastructure/
terraform apply -auto-approve
```

#### Step 2: Deploy Base Services (10 min)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install ingress-nginx
helm install nginx ingress-nginx/ingress-nginx
```

#### Step 3: Deploy LSTM-PFD (10 min)

```bash
helm install lstm-pfd ./deploy/helm/lstm-pfd \
  -f ./deploy/helm/lstm-pfd/values-prod.yaml \
  --set secrets.jwtSecret="$JWT_SECRET" \
  --set secrets.dbPassword="$DB_PASSWORD"
```

#### Step 4: Restore Data (15 min)

```bash
# Wait for PostgreSQL pod
kubectl wait --for=condition=Ready pod -l app=postgresql --timeout=300s

# Restore database
./scripts/disaster-recovery/restore-db.sh

# Restore models
aws s3 sync s3://lstm-pfd-backups/models/latest/ ./checkpoints/
```

#### Step 5: Verify System (5 min)

```bash
# Check all pods healthy
kubectl get pods

# Health check
curl https://lstm-pfd.your-domain.com/health

# Smoke test
curl -X POST https://lstm-pfd.your-domain.com/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [...]}'
```

---

## 4. Incident Response Checklist

### 4.1 Detection

- [ ] Alert received from PagerDuty/Prometheus
- [ ] Incident channel created (#incident-lstm-pfd-YYYYMMDD)
- [ ] On-call engineer acknowledged

### 4.2 Assessment

- [ ] Determine scope of impact
- [ ] Identify affected components
- [ ] Check recent deployments/changes
- [ ] Review error logs

### 4.3 Mitigation

- [ ] Execute appropriate runbook
- [ ] Monitor restoration progress
- [ ] Communicate status updates

### 4.4 Recovery Verification

- [ ] All health checks passing
- [ ] Metrics returning to baseline
- [ ] User functionality verified
- [ ] Customer communication sent

### 4.5 Post-Incident

- [ ] Incident timeline documented
- [ ] Root cause identified
- [ ] Action items created
- [ ] Post-mortem scheduled

---

## 5. Backup Schedule

| Backup Type       | Frequency     | Retention | Location                              |
| ----------------- | ------------- | --------- | ------------------------------------- |
| Database (full)   | Daily 2:00 AM | 30 days   | `s3://lstm-pfd-backups/db/`           |
| Database (WAL)    | Continuous    | 7 days    | `s3://lstm-pfd-backups/wal/`          |
| Model Checkpoints | Daily 3:00 AM | 30 days   | `s3://lstm-pfd-backups/models/`       |
| Configuration     | On change     | 90 days   | Git + `s3://lstm-pfd-backups/config/` |

---

## 6. Contact Information

| Role            | Contact                   | Escalation Time      |
| --------------- | ------------------------- | -------------------- |
| On-Call Primary | PagerDuty                 | Immediate            |
| Platform Lead   | platform-lead@example.com | 15 min               |
| Database Admin  | dba@example.com           | 30 min               |
| Security        | security@example.com      | If security incident |

---

## 7. Testing Schedule

DR procedures should be tested quarterly:

- **Q1:** Full cluster failover drill
- **Q2:** Database restoration test
- **Q3:** Model recovery test
- **Q4:** Full cluster failover drill

**Last Test Date:** TBD  
**Next Scheduled Test:** TBD

---

## Appendix: Common Issues

### A.1 Backup Download Fails

```bash
# Check AWS credentials
aws sts get-caller-identity

# Check bucket permissions
aws s3 ls s3://lstm-pfd-backups/
```

### A.2 PostgreSQL Won't Start

```bash
# Check pod logs
kubectl logs -l app=postgresql --tail=100

# Check PVC status
kubectl get pvc
```

### A.3 Model Won't Load

```bash
# Check model file exists
kubectl exec -it deployment/lstm-pfd-dashboard -- ls -la /app/checkpoints/

# Check model metrics
curl localhost:8050/metrics | grep model
```
