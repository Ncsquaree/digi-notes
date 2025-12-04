# AWS S3 and IAM Setup Guide for Digi Notes

This guide explains how to provision and configure AWS S3 and IAM for the Digi Notes application.

## Overview
- S3 is used to store uploaded note images.
- Backend generates presigned PUT URLs for frontend uploads.
- AI service downloads images from S3 for processing using secure credentials.

## Prerequisites
- AWS account
- AWS CLI (optional)

## Create S3 bucket
1. Open AWS Console → S3 → Create bucket.
2. Choose a unique bucket name: `digi-notes-uploads-{env}` (e.g. `digi-notes-uploads-prod`).
3. Region: match `AWS_REGION` in your env.
4. Disable public access (uploads/downloads via presigned URLs only).
5. Enable server-side encryption (SSE-S3 or SSE-KMS).
6. (Optional) Versioning and lifecycle rules to remove old temp files.

### CORS (if frontend uploads directly using presigned URLs)
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["PUT","POST","GET"],
    "AllowedOrigins": ["http://localhost:3000","http://localhost:19006"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3000
  }
]
```

## IAM user and policy
Create IAM user `digi-notes-app` and attach the following least-privilege policy (replace bucket name):

```json
{
  "Version":"2012-10-17",
  "Statement":[
    {"Effect":"Allow","Action":["s3:ListBucket","s3:GetBucketLocation"],"Resource":"arn:aws:s3:::digi-notes-uploads-prod"},
    {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject","s3:DeleteObject","s3:HeadObject"],"Resource":"arn:aws:s3:::digi-notes-uploads-prod/*"}
  ]
}
```

Store access keys securely (AWS Secrets Manager recommended).

## Environment variables
Set these in `backend/.env` and `ai/.env` (do not commit):

```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=digi-notes-uploads-prod
AWS_S3_PRESIGNED_URL_EXPIRY=3600
MAX_FILE_SIZE_MB=10
```

### AI service S3 settings
The AI service downloads files from S3 for processing. Add these to `ai/.env` (see `ai/.env.example`):

```
S3_DOWNLOAD_TIMEOUT=300            # seconds before download times out
S3_TEMP_DIR=/app/data/temp          # temporary directory inside the container for downloaded files
```

Recommendations:
- Use `S3_TEMP_DIR` inside the container (e.g. `/app/data/temp`) and ensure the directory is writable by the service user.
- Do not persist large temporary files across container restarts; clean up after processing.
- If you mount a volume for `S3_TEMP_DIR`, consider lifecycle rules to clear files older than a defined threshold.
- See `ai/.env.example` for defaults and adjust values per environment.

## Test access
Use AWS CLI to verify:

```bash
aws s3 ls s3://digi-notes-uploads-prod
aws s3 cp test.txt s3://digi-notes-uploads-prod/test.txt
aws s3 rm s3://digi-notes-uploads-prod/test.txt
```

## Production considerations
- Use Secrets Manager for credentials.
- Enable CloudTrail and S3 access logs.
- Use VPC endpoints for S3 if high security required.

## Troubleshooting
- AccessDenied: check IAM policy and bucket policy.
- SignatureDoesNotMatch: verify credentials and system clock.
