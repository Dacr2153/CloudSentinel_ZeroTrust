#!/usr/bin/env bash
# FILE: cloudsentinel-zero-trust/opensearch/setup/init_opensearch.sh
# ══════════════════════════════════════════════════════════════════════
# OpenSearch 2.11.x + Dashboards 2.11.x installer and configurator
# for CloudSentinel Zero-Trust SIEM on EC2 t2.micro (Amazon Linux 2023)
#
# Run context: EC2 UserData or manual SSH. Expects Amazon Linux 2023.
# Dependencies: curl, jq, systemd, awscli v2 (pre-installed on AL2023)
# ══════════════════════════════════════════════════════════════════════
set -euo pipefail
IFS=$'\n\t'

# ── Configuration ────────────────────────────────────────────────────
readonly OS_VERSION="2.11.1"
readonly OS_DASHBOARDS_VERSION="2.11.1"
readonly CLUSTER_NAME="cloudsentinel"
readonly JVM_HEAP="512m"   # 50% of t2.micro 1GB RAM
readonly DATA_DIR="/var/lib/opensearch"
readonly LOG_DIR="/var/log/opensearch"
readonly DASHBOARDS_PORT=5601
readonly OPENSEARCH_PORT=9200
readonly OPENSEARCH_PERF_PORT=9600

# SSM parameter prefix for password storage
readonly SSM_PREFIX="/cloudsentinel"

# Users to create
readonly ADMIN_USER="admin"
readonly LAMBDA_USER="cloudsentinel-lambda"
readonly READONLY_USER="cloudsentinel-readonly"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ── Helpers ──────────────────────────────────────────────────────────
log_info()    { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${BLUE}INFO${NC}  $*"; }
log_success() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}OK${NC}    $*"; }
log_warn()    { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}WARN${NC}  $*"; }
log_error()   { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}ERROR${NC} $*"; }

die() { log_error "$*"; exit 1; }

get_ec2_ip() {
    # IMDSv2 compliant — requires token
    local token
    token=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
    if [[ -n "$token" ]]; then
        curl -sH "X-aws-ec2-metadata-token: $token" \
            "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null || echo "localhost"
    else
        echo "localhost"
    fi
}

get_region() {
    local token
    token=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
    if [[ -n "$token" ]]; then
        curl -sH "X-aws-ec2-metadata-token: $token" \
            "http://169.254.169.254/latest/meta-data/placement/region" 2>/dev/null || echo "us-east-1"
    else
        echo "us-east-1"
    fi
}

generate_password() {
    # Generate a 24-char alphanumeric password with special chars
    local pw
    pw=$(openssl rand -base64 32 | tr -dc 'A-Za-z0-9!@#$%' | head -c 24)
    # Ensure complexity: at least 1 uppercase, 1 lowercase, 1 digit, 1 special
    echo "${pw}Aa1!"
}

store_ssm_parameter() {
    local name="$1"
    local value="$2"
    local region
    region=$(get_region)
    aws ssm put-parameter \
        --name "${SSM_PREFIX}/${name}" \
        --value "$value" \
        --type "SecureString" \
        --overwrite \
        --region "$region" \
        --no-cli-pager 2>/dev/null || log_warn "Failed to store SSM parameter: ${name}"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 1: System Prerequisites
# ══════════════════════════════════════════════════════════════════════
step1_prerequisites() {
    log_info "Step 1/8: Installing system prerequisites..."

    # Update system
    dnf update -y -q 2>/dev/null || yum update -y -q 2>/dev/null

    # Install required packages
    dnf install -y -q curl jq tar gzip unzip java-17-amazon-corretto-headless 2>/dev/null \
        || yum install -y -q curl jq tar gzip unzip java-17-amazon-corretto-headless 2>/dev/null

    # Verify Java
    java -version 2>&1 | head -1 || die "Java 17 installation failed"

    # Kernel tuning for OpenSearch
    # vm.max_map_count required by OpenSearch
    if ! grep -q "vm.max_map_count" /etc/sysctl.conf 2>/dev/null; then
        echo "vm.max_map_count=262144" >> /etc/sysctl.conf
    fi
    sysctl -w vm.max_map_count=262144 >/dev/null 2>&1

    # File descriptor limits
    cat > /etc/security/limits.d/opensearch.conf <<'LIMITS'
opensearch soft nofile 65536
opensearch hard nofile 65536
opensearch soft nproc 4096
opensearch hard nproc 4096
LIMITS

    # Create opensearch user if not exists
    id -u opensearch &>/dev/null || useradd -r -s /sbin/nologin opensearch

    log_success "System prerequisites installed"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Install OpenSearch 2.11.x
# ══════════════════════════════════════════════════════════════════════
step2_install_opensearch() {
    log_info "Step 2/8: Installing OpenSearch ${OS_VERSION}..."

    local os_tarball="opensearch-${OS_VERSION}-linux-x64.tar.gz"
    local os_url="https://artifacts.opensearch.org/releases/bundle/opensearch/${OS_VERSION}/${os_tarball}"

    if [[ -d "/opt/opensearch" ]]; then
        log_warn "OpenSearch directory already exists, skipping download"
    else
        cd /tmp
        log_info "Downloading OpenSearch ${OS_VERSION}..."
        curl -fSL -o "$os_tarball" "$os_url" || die "Failed to download OpenSearch"

        tar -xzf "$os_tarball" -C /opt/
        mv "/opt/opensearch-${OS_VERSION}" /opt/opensearch
        rm -f "$os_tarball"
    fi

    # Create data and log directories on EBS
    mkdir -p "$DATA_DIR" "$LOG_DIR"
    chown -R opensearch:opensearch /opt/opensearch "$DATA_DIR" "$LOG_DIR"

    log_success "OpenSearch ${OS_VERSION} installed to /opt/opensearch"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Configure OpenSearch
# ══════════════════════════════════════════════════════════════════════
step3_configure_opensearch() {
    log_info "Step 3/8: Configuring OpenSearch..."

    cat > /opt/opensearch/config/opensearch.yml <<YAML
# ── CloudSentinel OpenSearch Configuration ──
cluster.name: ${CLUSTER_NAME}
node.name: cloudsentinel-node-1
node.roles: [cluster_manager, data, ingest]

# Network — bound to all interfaces but protected by Security Group
network.host: 0.0.0.0
http.port: ${OPENSEARCH_PORT}
transport.port: 9300-9399

# Discovery — single-node setup (Free Tier)
discovery.type: single-node

# Paths — EBS mounted
path.data: ${DATA_DIR}
path.logs: ${LOG_DIR}

# Security plugin — disabled for internal VPC single-node
# In production, enable with mTLS and proper certificate management
plugins.security.disabled: true

# Performance — t2.micro optimized
indices.memory.index_buffer_size: 10%
thread_pool.write.queue_size: 500
action.auto_create_index: true

# Compatibility
compatibility.override_main_response_version: true
YAML

    # JVM settings — 50% of RAM (512MB for t2.micro)
    cat > /opt/opensearch/config/jvm.options.d/cloudsentinel.options <<JVM
-Xms${JVM_HEAP}
-Xmx${JVM_HEAP}
-XX:+UseG1GC
-XX:G1ReservePercent=25
-XX:InitiatingHeapOccupancyPercent=30
-XX:+HeapDumpOnOutOfMemoryError
-XX:HeapDumpPath=${LOG_DIR}/heap_dump.hprof
JVM

    chown -R opensearch:opensearch /opt/opensearch

    log_success "OpenSearch configured (cluster=${CLUSTER_NAME}, heap=${JVM_HEAP})"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Install OpenSearch Dashboards 2.11.x
# ══════════════════════════════════════════════════════════════════════
step4_install_dashboards() {
    log_info "Step 4/8: Installing OpenSearch Dashboards ${OS_DASHBOARDS_VERSION}..."

    local dash_tarball="opensearch-dashboards-${OS_DASHBOARDS_VERSION}-linux-x64.tar.gz"
    local dash_url="https://artifacts.opensearch.org/releases/bundle/opensearch-dashboards/${OS_DASHBOARDS_VERSION}/${dash_tarball}"

    if [[ -d "/opt/opensearch-dashboards" ]]; then
        log_warn "Dashboards directory already exists, skipping download"
    else
        cd /tmp
        log_info "Downloading OpenSearch Dashboards ${OS_DASHBOARDS_VERSION}..."
        curl -fSL -o "$dash_tarball" "$dash_url" || die "Failed to download Dashboards"

        tar -xzf "$dash_tarball" -C /opt/
        mv "/opt/opensearch-dashboards-${OS_DASHBOARDS_VERSION}" /opt/opensearch-dashboards
        rm -f "$dash_tarball"
    fi

    # Configure dashboards
    cat > /opt/opensearch-dashboards/config/opensearch_dashboards.yml <<YAML
# ── CloudSentinel Dashboards Configuration ──
server.host: "0.0.0.0"
server.port: ${DASHBOARDS_PORT}
server.name: "cloudsentinel-dashboards"

# Point to local OpenSearch
opensearch.hosts: ["http://localhost:${OPENSEARCH_PORT}"]

# Disable security plugin (matching OpenSearch config)
opensearch_security.multitenancy.enabled: false
opensearch_security.readonly_mode.roles: ["kibana_read_only"]

# Logging
logging.dest: /var/log/opensearch/dashboards.log
logging.verbose: false

# Default index pattern
opensearch.requestHeadersAllowlist: ["securitytenant", "Authorization"]
YAML

    chown -R opensearch:opensearch /opt/opensearch-dashboards

    log_success "OpenSearch Dashboards ${OS_DASHBOARDS_VERSION} installed"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Create systemd Services
# ══════════════════════════════════════════════════════════════════════
step5_create_services() {
    log_info "Step 5/8: Creating systemd services..."

    # OpenSearch service
    cat > /etc/systemd/system/opensearch.service <<'UNIT'
[Unit]
Description=OpenSearch 2.11.x — CloudSentinel SIEM
Documentation=https://opensearch.org/docs/latest/
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=opensearch
Group=opensearch
RuntimeDirectory=opensearch
Environment=OPENSEARCH_HOME=/opt/opensearch
Environment=OPENSEARCH_PATH_CONF=/opt/opensearch/config
WorkingDirectory=/opt/opensearch

ExecStart=/opt/opensearch/bin/opensearch

LimitNOFILE=65536
LimitNPROC=4096
LimitAS=infinity
LimitFSIZE=infinity

# Restart policy
Restart=on-failure
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Security hardening
NoNewPrivileges=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=/var/lib/opensearch /var/log/opensearch /opt/opensearch

[Install]
WantedBy=multi-user.target
UNIT

    # Dashboards service
    cat > /etc/systemd/system/opensearch-dashboards.service <<'UNIT'
[Unit]
Description=OpenSearch Dashboards 2.11.x — CloudSentinel UI
Documentation=https://opensearch.org/docs/latest/dashboards/
After=opensearch.service
Requires=opensearch.service

[Service]
Type=simple
User=opensearch
Group=opensearch
Environment=NODE_OPTIONS="--max-old-space-size=256"
WorkingDirectory=/opt/opensearch-dashboards

ExecStart=/opt/opensearch-dashboards/bin/opensearch-dashboards

LimitNOFILE=65536

Restart=on-failure
RestartSec=15
StartLimitInterval=120
StartLimitBurst=3

NoNewPrivileges=true
ProtectHome=true

[Install]
WantedBy=multi-user.target
UNIT

    # Reload and enable
    systemctl daemon-reload
    systemctl enable opensearch.service
    systemctl enable opensearch-dashboards.service

    log_success "systemd services created and enabled"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 6: Start Services + Health Check Loop
# ══════════════════════════════════════════════════════════════════════
step6_start_and_healthcheck() {
    log_info "Step 6/8: Starting OpenSearch and performing health check..."

    systemctl start opensearch.service

    # Health check loop — wait up to 5 minutes for cluster green/yellow
    local max_wait=300
    local interval=10
    local elapsed=0

    log_info "Waiting for OpenSearch to become healthy (max ${max_wait}s)..."

    while [[ $elapsed -lt $max_wait ]]; do
        local status
        status=$(curl -sf "http://localhost:${OPENSEARCH_PORT}/_cluster/health" 2>/dev/null \
            | jq -r '.status' 2>/dev/null || echo "unreachable")

        if [[ "$status" == "green" || "$status" == "yellow" ]]; then
            log_success "OpenSearch cluster health: ${status} (${elapsed}s)"
            break
        fi

        log_info "  Cluster status: ${status} — waiting... (${elapsed}s/${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    if [[ $elapsed -ge $max_wait ]]; then
        die "OpenSearch did not become healthy within ${max_wait}s"
    fi

    # Print cluster info
    log_info "Cluster info:"
    curl -sf "http://localhost:${OPENSEARCH_PORT}" 2>/dev/null | jq . || true

    # Start Dashboards
    systemctl start opensearch-dashboards.service
    sleep 10

    # Verify Dashboards is responding
    local dash_status
    dash_status=$(curl -sf -o /dev/null -w "%{http_code}" \
        "http://localhost:${DASHBOARDS_PORT}/api/status" 2>/dev/null || echo "000")

    if [[ "$dash_status" == "200" ]]; then
        log_success "OpenSearch Dashboards is running on port ${DASHBOARDS_PORT}"
    else
        log_warn "Dashboards may still be starting (HTTP ${dash_status}). Check manually."
    fi
}

# ══════════════════════════════════════════════════════════════════════
# STEP 7: Create Index Templates
# ══════════════════════════════════════════════════════════════════════
step7_create_index_templates() {
    log_info "Step 7/8: Creating index templates..."

    local os_url="http://localhost:${OPENSEARCH_PORT}"

    # ── Events index template ────────────────────────────────────────
    log_info "Creating cloudsentinel-events index template..."
    local events_mapping
    events_mapping=$(cat "$(dirname "$0")/../indices/cloudtrail-events-mapping.json" 2>/dev/null || echo "")

    if [[ -n "$events_mapping" ]]; then
        curl -sf -X PUT "${os_url}/_index_template/cloudsentinel-events" \
            -H "Content-Type: application/json" \
            -d "$events_mapping" | jq . 2>/dev/null
        log_success "Events index template created"
    else
        log_warn "cloudtrail-events-mapping.json not found — skipping"
    fi

    # ── Alerts index template ────────────────────────────────────────
    log_info "Creating cloudsentinel-alerts index template..."
    local alerts_mapping
    alerts_mapping=$(cat "$(dirname "$0")/../indices/alerts-mapping.json" 2>/dev/null || echo "")

    if [[ -n "$alerts_mapping" ]]; then
        curl -sf -X PUT "${os_url}/_index_template/cloudsentinel-alerts" \
            -H "Content-Type: application/json" \
            -d "$alerts_mapping" | jq . 2>/dev/null
        log_success "Alerts index template created"
    else
        log_warn "alerts-mapping.json not found — skipping"
    fi

    # ── ILM Policies ─────────────────────────────────────────────────
    log_info "Creating ILM (ISM) policies..."

    # Events ISM policy: hot (7d) → warm (23d) → delete (90d)
    curl -sf -X PUT "${os_url}/_plugins/_ism/policies/cloudsentinel-events-lifecycle" \
        -H "Content-Type: application/json" \
        -d '{
  "policy": {
    "description": "CloudSentinel events index lifecycle — hot 7d, warm 23d, delete 90d",
    "default_state": "hot",
    "states": [
      {
        "name": "hot",
        "actions": [
          {
            "rollover": {
              "min_index_age": "7d",
              "min_primary_shard_size": "10gb"
            }
          }
        ],
        "transitions": [
          { "state_name": "warm", "conditions": { "min_index_age": "7d" } }
        ]
      },
      {
        "name": "warm",
        "actions": [
          { "force_merge": { "max_num_segments": 1 } },
          { "read_only": {} }
        ],
        "transitions": [
          { "state_name": "delete", "conditions": { "min_index_age": "90d" } }
        ]
      },
      {
        "name": "delete",
        "actions": [{ "delete": {} }],
        "transitions": []
      }
    ],
    "ism_template": {
      "index_patterns": ["cloudsentinel-events-*"],
      "priority": 100
    }
  }
}' | jq . 2>/dev/null || log_warn "Events ISM policy creation failed (may already exist)"
    log_success "Events ISM policy created"

    # Alerts ISM policy: hot (30d) → warm (335d) → delete (365d, 1 year)
    curl -sf -X PUT "${os_url}/_plugins/_ism/policies/cloudsentinel-alerts-lifecycle" \
        -H "Content-Type: application/json" \
        -d '{
  "policy": {
    "description": "CloudSentinel alerts retention — 1 year",
    "default_state": "hot",
    "states": [
      {
        "name": "hot",
        "actions": [],
        "transitions": [
          { "state_name": "warm", "conditions": { "min_index_age": "30d" } }
        ]
      },
      {
        "name": "warm",
        "actions": [
          { "force_merge": { "max_num_segments": 1 } },
          { "read_only": {} }
        ],
        "transitions": [
          { "state_name": "delete", "conditions": { "min_index_age": "365d" } }
        ]
      },
      {
        "name": "delete",
        "actions": [{ "delete": {} }],
        "transitions": []
      }
    ],
    "ism_template": {
      "index_patterns": ["cloudsentinel-alerts-*"],
      "priority": 100
    }
  }
}' | jq . 2>/dev/null || log_warn "Alerts ISM policy creation failed (may already exist)"
    log_success "Alerts ISM policy created"

    # ── Create initial indices ───────────────────────────────────────
    local today
    today=$(date +%Y.%m.%d)

    curl -sf -X PUT "${os_url}/cloudsentinel-events-${today}" \
        -H "Content-Type: application/json" \
        -d '{"settings": {"number_of_shards": 1, "number_of_replicas": 0}}' \
        | jq . 2>/dev/null || log_warn "Initial events index may already exist"

    curl -sf -X PUT "${os_url}/cloudsentinel-alerts-${today}" \
        -H "Content-Type: application/json" \
        -d '{"settings": {"number_of_shards": 1, "number_of_replicas": 0}}' \
        | jq . 2>/dev/null || log_warn "Initial alerts index may already exist"

    log_success "Initial indices created for ${today}"
}

# ══════════════════════════════════════════════════════════════════════
# STEP 8: Store Credentials in SSM + Summary
# ══════════════════════════════════════════════════════════════════════
step8_finalize() {
    log_info "Step 8/8: Storing configuration in SSM and generating summary..."

    local ec2_ip
    ec2_ip=$(get_ec2_ip)
    local region
    region=$(get_region)

    # Generate passwords
    local admin_pw lambda_pw readonly_pw
    admin_pw=$(generate_password)
    lambda_pw=$(generate_password)
    readonly_pw=$(generate_password)

    # Store in SSM Parameter Store
    store_ssm_parameter "opensearch/admin-password" "$admin_pw"
    store_ssm_parameter "opensearch/lambda-password" "$lambda_pw"
    store_ssm_parameter "opensearch/readonly-password" "$readonly_pw"
    store_ssm_parameter "opensearch/endpoint" "http://${ec2_ip}:${OPENSEARCH_PORT}"
    store_ssm_parameter "opensearch/dashboards-endpoint" "http://${ec2_ip}:${DASHBOARDS_PORT}"

    log_success "Credentials stored in SSM Parameter Store (${SSM_PREFIX}/opensearch/*)"

    # ── Summary ──────────────────────────────────────────────────────
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✅ OpenSearch Setup Complete — CloudSentinel Zero-Trust${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  OpenSearch:    ${BLUE}http://${ec2_ip}:${OPENSEARCH_PORT}${NC}"
    echo -e "  Dashboards:    ${BLUE}http://${ec2_ip}:${DASHBOARDS_PORT}${NC}"
    echo -e "  Cluster Name:  ${CLUSTER_NAME}"
    echo -e "  Version:       ${OS_VERSION}"
    echo -e "  JVM Heap:      ${JVM_HEAP}"
    echo -e "  Data Dir:      ${DATA_DIR}"
    echo -e "  Region:        ${region}"
    echo ""
    echo -e "  SSM Parameters:"
    echo -e "    ${SSM_PREFIX}/opensearch/admin-password"
    echo -e "    ${SSM_PREFIX}/opensearch/lambda-password"
    echo -e "    ${SSM_PREFIX}/opensearch/readonly-password"
    echo -e "    ${SSM_PREFIX}/opensearch/endpoint"
    echo -e "    ${SSM_PREFIX}/opensearch/dashboards-endpoint"
    echo ""
    echo -e "  ${YELLOW}Note: Security plugin is disabled (internal VPC only).${NC}"
    echo -e "  ${YELLOW}For production, enable mTLS with proper certificates.${NC}"
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}"
}

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
main() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  CloudSentinel Zero-Trust — OpenSearch Installer${NC}"
    echo -e "${BLUE}  OpenSearch ${OS_VERSION} + Dashboards ${OS_DASHBOARDS_VERSION}${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo ""

    local start_time
    start_time=$(date +%s)

    step1_prerequisites
    step2_install_opensearch
    step3_configure_opensearch
    step4_install_dashboards
    step5_create_services
    step6_start_and_healthcheck
    step7_create_index_templates
    step8_finalize

    local end_time duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    log_success "Total setup time: ${duration}s"
}

main "$@"
