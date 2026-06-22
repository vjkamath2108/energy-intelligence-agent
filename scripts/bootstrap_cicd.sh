#!/usr/bin/env bash
# =============================================================================
# bootstrap_cicd.sh — One-time CI/CD setup for energy-intelligence-agent
#
# Run ONCE from a machine with AWS credentials. Creates:
#   1. GitHub OIDC provider in IAM (shared across projects — skipped if exists)
#   2. IAM role  energy-agent-github-ci  trusted by GitHub Actions, with
#      least-privilege Bedrock permissions for integration tests
#
# The role is intentionally scoped to Bedrock only — no RDS, no S3.
# Integration tests use a GitHub Actions PostgreSQL service container (free,
# ephemeral) rather than a persistent RDS instance.
#
# Usage:
#   bash scripts/bootstrap_cicd.sh --profile vijay-dev
#
# After running, add these GitHub repository secrets:
#   AWS_ACCOUNT_ID   (printed below)
#   AWS_ROLE_ARN     (printed below)
# =============================================================================

set -euo pipefail

GITHUB_REPO="vjkamath2108/energy-intelligence-agent"
AWS_PROFILE_ARG=""
REGION="eu-west-2"
ROLE_NAME="energy-agent-github-ci"
OIDC_URL="https://token.actions.githubusercontent.com"
OIDC_THUMBPRINT="6938fd4d98bab03faadb97b34396831e3780aea1"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC}  $1"; }
info() { echo -e "  ${BLUE}→${NC}  $1"; }
warn() { echo -e "  ${YELLOW}!${NC}  $1"; }
fail() { echo -e "  ${RED}✗${NC}  $1"; exit 1; }
step() { echo -e "\n${BOLD}━━━  $1  ━━━${NC}\n"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) AWS_PROFILE_ARG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$AWS_PROFILE_ARG" ]] && export AWS_PROFILE="$AWS_PROFILE_ARG"
PROFILE_FLAG="${AWS_PROFILE_ARG:+--profile $AWS_PROFILE_ARG}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
aws() { command aws $PROFILE_FLAG "$@"; }

# =============================================================================
step "Preflight"
# =============================================================================
command -v aws jq &>/dev/null || fail "aws and jq are required (brew install awscli jq)"
IDENTITY=$(command aws $PROFILE_FLAG sts get-caller-identity --output json) \
  || fail "AWS credentials not found — run 'aws configure' or check --profile"
ACCOUNT_ID=$(echo "$IDENTITY" | jq -r '.Account')
CALLER=$(echo "$IDENTITY" | jq -r '.Arn')
ok "Account: $ACCOUNT_ID  Region: $REGION"
ok "Caller:  $CALLER"

# =============================================================================
step "1 / 2 — GitHub OIDC Provider"
# =============================================================================
EXISTING_OIDC=$(aws iam list-open-id-connect-providers \
  --query "OpenIDConnectProviderList[?ends_with(Arn, '/token.actions.githubusercontent.com')].Arn" \
  --output text 2>/dev/null || true)

if [[ -n "$EXISTING_OIDC" ]]; then
  ok "Already exists (shared across projects): $EXISTING_OIDC"
  OIDC_ARN="$EXISTING_OIDC"
else
  OIDC_ARN=$(aws iam create-open-id-connect-provider \
    --url "$OIDC_URL" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "$OIDC_THUMBPRINT" \
    --query 'OpenIDConnectProviderArn' --output text)
  ok "Created: $OIDC_ARN"
fi

# =============================================================================
step "2 / 2 — IAM Role: $ROLE_NAME"
# =============================================================================

# Trust policy: only this repo's GitHub Actions workflows can assume the role.
# StringLike on :sub allows any ref (branch, tag, PR) from this repo.
TRUST=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "GitHubActionsOIDC",
    "Effect": "Allow",
    "Principal": { "Federated": "${OIDC_ARN}" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:${GITHUB_REPO}:*"
      }
    }
  }]
}
EOF
)

# Least-privilege permissions:
#   bedrock:InvokeModel   — Claude Sonnet (recommendation engine) and
#                           Titan Embed V2 (hybrid retriever embeddings)
#   bedrock:ListFoundationModels — discovery / smoke test
# No RDS permissions — integration tests use a GitHub Actions service container.
PERMISSIONS=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInferenceModels",
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": [
        "arn:aws:bedrock:${REGION}::foundation-model/anthropic.claude-sonnet-4-5",
        "arn:aws:bedrock:${REGION}::foundation-model/amazon.titan-embed-text-v2:0"
      ]
    },
    {
      "Sid": "BedrockDiscovery",
      "Effect": "Allow",
      "Action": "bedrock:ListFoundationModels",
      "Resource": "*"
    }
  ]
}
EOF
)

EXISTING_ROLE=$(aws iam get-role --role-name "$ROLE_NAME" \
  --query 'Role.RoleName' --output text 2>/dev/null || true)

if [[ -n "$EXISTING_ROLE" ]]; then
  info "Role already exists — updating trust policy."
  aws iam update-assume-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-document "$TRUST"
  ok "Trust policy updated."
else
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST" \
    --description "GitHub Actions OIDC role for energy-intelligence-agent CI — Bedrock only" \
    --max-session-duration 3600 \
    --no-cli-pager
  ok "Role created."
fi

# Put (create or replace) the inline permissions policy.
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name "BedrockIntegrationTests" \
  --policy-document "$PERMISSIONS"
ok "Inline policy 'BedrockIntegrationTests' applied (Bedrock invoke + list)."

ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
ok "Role ARN: $ROLE_ARN"

# =============================================================================
cat <<SECRETS

${BOLD}━━━  Add these GitHub repository secrets  ━━━${NC}
  https://github.com/${GITHUB_REPO}/settings/secrets/actions

  AWS_ACCOUNT_ID   =  ${ACCOUNT_ID}
  AWS_ROLE_ARN     =  ${ROLE_ARN}

${BOLD}━━━  What these enable in CI  ━━━${NC}
  • integration-test job in .github/workflows/ci.yml
  • Runs pytest -m integration against real Bedrock (Claude Sonnet + Titan Embed V2)
  • PostgreSQL comes from a GitHub Actions service container — no RDS needed
  • Job runs on push to main only (not on PRs from forks)

${BOLD}━━━  Bedrock model availability check  ━━━${NC}
  Verify the models are enabled in ${REGION}:
    aws bedrock list-foundation-models --region ${REGION} --profile ${AWS_PROFILE_ARG:-default} \\
      --query 'modelSummaries[?contains(modelId, \`claude-sonnet-4-5\`) || contains(modelId, \`titan-embed-text-v2\`)].modelId'

SECRETS
