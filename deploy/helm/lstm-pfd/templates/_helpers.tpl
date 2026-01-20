{{/*
Expand the name of the chart.
*/}}
{{- define "lstm-pfd.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "lstm-pfd.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "lstm-pfd.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "lstm-pfd.labels" -}}
helm.sh/chart: {{ include "lstm-pfd.chart" . }}
{{ include "lstm-pfd.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "lstm-pfd.selectorLabels" -}}
app.kubernetes.io/name: {{ include "lstm-pfd.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "lstm-pfd.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "lstm-pfd.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Dashboard component labels
*/}}
{{- define "lstm-pfd.dashboard.labels" -}}
{{ include "lstm-pfd.labels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
Dashboard selector labels
*/}}
{{- define "lstm-pfd.dashboard.selectorLabels" -}}
{{ include "lstm-pfd.selectorLabels" . }}
app.kubernetes.io/component: dashboard
{{- end }}

{{/*
Worker component labels
*/}}
{{- define "lstm-pfd.worker.labels" -}}
{{ include "lstm-pfd.labels" . }}
app.kubernetes.io/component: worker
{{- end }}

{{/*
Worker selector labels
*/}}
{{- define "lstm-pfd.worker.selectorLabels" -}}
{{ include "lstm-pfd.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end }}
