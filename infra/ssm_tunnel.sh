#!/usr/bin/env bash
# ssm_tunnel.sh — forward localhost:5433 → RDS via SSM port-forwarding through the tunnel EC2
#
# Usage:
#   ./ssm_tunnel.sh                    # uses defaults from terraform output
#   INSTANCE_ID=i-xxx ./ssm_tunnel.sh  # override instance
#
# Prerequisites:
#   - AWS CLI v2  (https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
#   - Session Manager plugin  (https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
#   - Terraform applied:  cd infra && terraform apply
#   - AWS profile set:    export AWS_PROFILE=vijay-dev  (or pass --profile below)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="eu-west-2"  # infra is in eu-west-2; ignore AWS_REGION env var
PROFILE="${AWS_PROFILE:-}"
LOCAL_PORT="${LOCAL_PORT:-5433}"

# Known-good fallbacks — override with env vars or terraform output
DEFAULT_INSTANCE_ID="i-060ad8218a92ddeb9"
DEFAULT_RDS_HOST="energy-intelligence-agent-postgres.chg2ymciis6q.eu-west-2.rds.amazonaws.com"

# ── Resolve outputs from Terraform (falls back to hardcoded defaults) ───────
if [[ -z "${INSTANCE_ID:-}" ]]; then
  INSTANCE_ID=$(terraform -chdir="$SCRIPT_DIR" output -raw ec2_instance_id 2>/dev/null) || {
    echo "Terraform state unavailable — using default instance ID."
    INSTANCE_ID="$DEFAULT_INSTANCE_ID"
  }
fi

if [[ -z "${RDS_HOST:-}" ]]; then
  RDS_HOST=$(terraform -chdir="$SCRIPT_DIR" output -raw rds_address 2>/dev/null) || {
    echo "Terraform state unavailable — using default RDS host."
    RDS_HOST="$DEFAULT_RDS_HOST"
  }
fi

RDS_PORT="${RDS_PORT:-5432}"

PROFILE_ARG=()
[[ -n "${PROFILE:-}" ]] && PROFILE_ARG=(--profile "$PROFILE")

# ── Resolve effective AWS identity ─────────────────────────────────────────
EFFECTIVE_PROFILE="${PROFILE:-<env/instance-profile>}"
CALLER_IDENTITY=$(aws sts get-caller-identity "${PROFILE_ARG[@]}" --region "$REGION" --output text --query 'Arn' 2>/dev/null) \
  || CALLER_IDENTITY="(could not resolve — check credentials)"

echo ""
echo "  AWS profile : $EFFECTIVE_PROFILE"
echo "  AWS identity: $CALLER_IDENTITY"
echo "  Region      : $REGION"
echo ""
echo "  Instance : $INSTANCE_ID"
echo "  RDS host : $RDS_HOST:$RDS_PORT"
echo "  Tunnel   : localhost:$LOCAL_PORT -> $RDS_HOST:$RDS_PORT"
echo ""

# ── Pre-flight: check SSM agent is registered ──────────────────────────────
echo "  Checking SSM connectivity..."
SSM_STATUS=$(aws ssm describe-instance-information "${PROFILE_ARG[@]}" \
  --region "$REGION" \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --query 'InstanceInformationList[0].PingStatus' \
  --output text 2>/dev/null) || SSM_STATUS="Unknown"

if [[ "$SSM_STATUS" != "Online" ]]; then
  echo ""
  echo "  WARNING: SSM agent ping status = '$SSM_STATUS' (expected 'Online')"
  echo "  The tunnel may fail. Check that the SSM agent is running on the instance."
  echo "  Wait 30-60s after an EC2 start for the agent to register, then retry."
  echo ""
fi

echo "  SSM status  : $SSM_STATUS"
echo ""
echo "  Connect with:"
echo "    psql -h localhost -p $LOCAL_PORT -U <db_username> -d energy_db"
echo "    or set DB_HOST=localhost DB_PORT=$LOCAL_PORT in your .env"
echo ""
echo "  Press Ctrl-C to close the tunnel."
echo ""

aws ssm start-session \
  "${PROFILE_ARG[@]}" \
  --region  "$REGION" \
  --target  "$INSTANCE_ID" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters "{\"host\":[\"$RDS_HOST\"],\"portNumber\":[\"$RDS_PORT\"],\"localPortNumber\":[\"$LOCAL_PORT\"]}"
