{{/*
Expand the name of the chart.
*/}}
{{- define "agentblackbox.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "agentblackbox.fullname" -}}
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
Create chart label.
*/}}
{{- define "agentblackbox.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "agentblackbox.labels" -}}
helm.sh/chart: {{ include "agentblackbox.chart" . }}
app.kubernetes.io/name: {{ include "agentblackbox.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "agentblackbox.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentblackbox.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Database URL helper.
*/}}
{{- define "agentblackbox.databaseUrl" -}}
{{- printf "postgresql+asyncpg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@%s-postgres:%d/agentblackbox" (include "agentblackbox.fullname" .) (.Values.postgres.service.port | int) }}
{{- end }}
