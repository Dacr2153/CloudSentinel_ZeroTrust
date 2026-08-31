#!/usr/bin/env bash
# =============================================================================
# CloudSentinel Zero-Trust — LOCAL Pipeline Setup & Validation Script
#
# Trains the Isolation Forest model on synthetic labeled data (standard ML
# practice) and validates the pipeline is ready to process REAL CloudTrail
# events exported from AWS.
#
# Usage:
#   ./scripts/local_demo.sh [--skip-training]
#
# To process events after setup:
#   export $(grep -v '^#' .env.local | xargs)
#   python cloudsentinel.py --no-menu --run-pipeline
#
# Obtaining real CloudTrail events:
#   AWS Console → CloudTrail → Event history → Download JSON
#   Place .json or .json.gz files in: ml/data/cloudtrail_samples/
#
# Requirements:
#   python3 -m venv .venv && pip install -r requirements-dev.txt
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SKIP_TRAINING=false

for arg in "$@"; do
  case "$arg" in
    --skip-training) SKIP_TRAINING=true ;;
    --help|-h)
      echo "Usage: $0 [--skip-training]"
      exit 0
      ;;
  esac
done

echo ""
echo -e "${BOLD}${BLUE}============================================================${NC}"
echo -e "${BOLD}${BLUE}  CloudSentinel Zero-Trust — Pipeline Setup${NC}"
echo -e "${BOLD}${BLUE}============================================================${NC}"
echo ""

# ── Step 0: Activate venv ──
if [[ ! -f ".venv/bin/activate" ]]; then
  echo -e "${RED}ERROR: .venv not found. Run 'python3 -m venv .venv && pip install -r requirements-dev.txt' first.${NC}"
  exit 1
fi
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ── Step 1: Load env ──
if [[ -f ".env.local" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' .env.local | grep -v '^$' | xargs)
  echo -e "${GREEN}✓ .env.local loaded${NC}"
else
  echo -e "${YELLOW}⚠ .env.local not found, using defaults${NC}"
  export CLOUDSENTINEL_LOCAL_MODE=true
  export AWS_DEFAULT_REGION=us-east-1
fi

# ── Step 2: Create directories ──
echo ""
echo -e "${CYAN}[1/3] Setting up directories...${NC}"
mkdir -p ml/data/local/models/isolation_forest
mkdir -p ml/data/cloudtrail_samples
mkdir -p tools/alerts
mkdir -p reports
echo -e "${GREEN}  ✓ Directories ready${NC}"

# ── Step 3: Generate synthetic training data + train model ──
if [[ "$SKIP_TRAINING" == "false" ]]; then
  echo ""
  echo -e "${CYAN}[2/3] Generating synthetic training data (labeled feature vectors)...${NC}"
  echo -e "  ${YELLOW}Note: Synthetic data is used ONLY for training the Isolation Forest model.${NC}"
  echo -e "  ${YELLOW}The pipeline detects anomalies in REAL CloudTrail events you provide.${NC}"
  python ml/training/generate_synthetic_data.py --output-dir ml/data
  echo -e "${GREEN}  ✓ Training data generated${NC}"

  echo ""
  echo -e "${CYAN}[3/3] Training Isolation Forest anomaly detection model...${NC}"
  python ml/training/train_model.py \
    --data-dir ml/data \
    --output-dir ml/data
  echo -e "${GREEN}  ✓ Model trained and saved${NC}"
else
  echo ""
  echo -e "${YELLOW}[2/3] Skipping training (--skip-training)${NC}"
  echo -e "${YELLOW}[3/3] Skipping training (--skip-training)${NC}"
fi

# ── Verify model exists ──
MODEL_PATH="ml/data/local/models/isolation_forest/model.joblib"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo -e "${RED}ERROR: Model not found at $MODEL_PATH${NC}"
  echo "  Run without --skip-training to generate the model."
  exit 1
fi
echo -e "${GREEN}  ✓ Model ready: $MODEL_PATH${NC}"

# ── Summary ──
echo ""
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo -e "${BOLD}${GREEN}  Setup complete! Pipeline is ready.${NC}"
echo -e "${BOLD}${GREEN}============================================================${NC}"
echo ""
echo -e "  To analyze your AWS CloudTrail logs:"
echo -e "  ${CYAN}1. Export events from AWS:${NC}"
echo -e "     AWS Console → CloudTrail → Event history → Download JSON"
echo -e "     (or download .json.gz files from your S3 CloudTrail bucket)"
echo ""
echo -e "  ${CYAN}2. Place the exported files in:${NC}"
echo -e "     ml/data/cloudtrail_samples/"
echo ""
echo -e "  ${CYAN}3. Run the detection pipeline:${NC}"
echo -e "     export \$(grep -v '^#' .env.local | xargs)"
echo -e "     python cloudsentinel.py --no-menu --run-pipeline"
echo ""
echo -e "  ${CYAN}Or use the interactive menu:${NC}"
echo -e "     python cloudsentinel.py"
echo ""
echo -e "  ${CYAN}Other commands:${NC}"
echo -e "     make test              # Run unit tests"
echo -e "     make evaluate          # Evaluate model performance"
echo -e "     python cloudsentinel.py --help"
echo ""
