# Multi-stage build: Extract Triton from NVIDIA's official image
FROM nvcr.io/nvidia/tritonserver:23.12-py3 AS triton-source

# Prepare Triton Python bindings in source stage
RUN mkdir -p /tmp/triton-bindings && \
    if [ -d "/usr/local/lib/python3.10/dist-packages" ]; then \
        find /usr/local/lib/python3.10/dist-packages -name "tritonserver*" -exec cp -r {} /tmp/triton-bindings/ \; 2>/dev/null || true; \
    fi && \
    if [ -d "/opt/tritonserver/python" ]; then \
        cp -r /opt/tritonserver/python/* /tmp/triton-bindings/ 2>/dev/null || true; \
    fi

# Main image: Ubuntu 24.04 with Python 3.10 (to match Triton's Python version)
FROM ubuntu:24.04

# Set noninteractive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Define build arguments from GitHub Actions workflow
ARG KERNEL_VERSION
ARG DISABLE_AUTOLOGIN
ARG NVIDIA_INSTALL_ENABLED
ARG NVIDIA_DRIVER_URL
ARG FIRSTBOOT_ENABLED
ARG KERNEL_INSTALL_ENABLED

# =============================================================================
# USER & GROUP SETUP - Digital Research Alliance of Canada (DRAC)
# =============================================================================

# --- 0. Set root user ---
USER root

# --- 1. Set root password ---
RUN echo "root:changeme" | chpasswd

# --- 2. Create wwuser user accounts (UID 2000) ---
RUN mkdir -p /local/home && \
    groupadd -g 2000 wwgroup && \
    useradd -u 2000 -m -d /local/home/wwuser -g wwgroup -G sudo -s /bin/bash wwuser && \
    echo "wwuser:wwpassword" | chpasswd

# --- 3. Create distributive.network user (UID 2001) ---
RUN groupadd -g 2001 distgroup && \
    useradd -u 2001 -m -d /local/home/dist -g distgroup -s /bin/bash dist

# --- 4. Update package lists (with error handling for mirror sync issues) ---
# Handle transient mirror sync errors gracefully (dep11 metadata issues)
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false || \
    (sleep 5 && apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false) || true

# --- 5. Install Core Tools, Debugging, and Dependencies ---
RUN apt-get install -y \
    # Essential system
    sudo \
    openssh-server \
    openssh-client \
    systemd \
    systemd-sysv \
    dbus \
    rsyslog \
    # Essential networking
    iproute2 \
    iputils-ping \
    netplan.io \
    # Essential utilities
    curl \
    wget \
    less \
    unzip \
    # Testing/debugging tools
    vim \
    htop \
    strace \
    lsof \
    tcpdump \
    jq \
    git \
    # Essential for Ray/Python
    python3 \
    python3-pip \
    python3-venv \
    # Essential for HPC/compute
    numactl \
    cgroup-tools \
    kmod \
    # Storage (if needed)
    nfs-common \
    multipath-tools \
    # Essential dependencies
    ca-certificates \
    libnuma1 \
    libpam0g \
    libyaml-0-2 \
    libjson-c5 \
    libssl3 \
    libcurl4 \
    libdbus-1-3 \
    netbase && \
    # Optional: Ansible (only if firstboot enabled)
    if [ "$FIRSTBOOT_ENABLED" = "true" ]; then \
        apt-get install -y ansible; \
    fi && \
    if [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        apt-get install -y \
            linux-image-${KERNEL_VERSION} \
            linux-headers-${KERNEL_VERSION} \
            linux-modules-${KERNEL_VERSION} \
            linux-modules-extra-${KERNEL_VERSION} \
            initramfs-tools && \
        ln -s /usr/src/linux-headers-${KERNEL_VERSION} /lib/modules/${KERNEL_VERSION}/build; \
    fi && \
    if [ "$NVIDIA_INSTALL_ENABLED" = "true" ] && [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        apt-get install -y \
            build-essential \
            pkg-config \
            xorg-dev \
            libx11-dev \
            libxext-dev \
            libglvnd-dev \
            libglvnd-core-dev \
            libgl-dev \
            libegl-dev \
            libgles-dev; \
    fi && \
    mkdir -p /var/log/journal && \
    systemd-tmpfiles --create --prefix /var/log/journal && \
    systemctl mask \
      systemd-udevd.service \
      systemd-udevd-kernel.socket \
      systemd-udevd-control.socket \
      systemd-modules-load.service \
      sys-kernel-config.mount \
      sys-kernel-debug.mount \
      sys-fs-fuse-connections.mount \
      systemd-remount-fs.service \
      getty.target \
      systemd-logind.service \
      systemd-vconsole-setup.service \
      systemd-timesyncd.service

# --- 6. Fetch and Apply SCAP Security Guide Remediation (optional) ---
# Install SCAP scanner and apply CIS Level 2 Server profile remediation
RUN apt-get install -y openscap-scanner libopenscap25t64 && \
    export SSG_VERSION=$(curl -s https://api.github.com/repos/ComplianceAsCode/content/releases/latest | grep -oP '"tag_name": "\K[^"]+' || echo "0.1.66") && \
    echo "🔄 Using SCAP Security Guide version: $SSG_VERSION" && \
    SSG_VERSION_NO_V=$(echo "$SSG_VERSION" | sed 's/^v//') && \
    wget -O /ssg.zip "https://github.com/ComplianceAsCode/content/releases/download/${SSG_VERSION}/scap-security-guide-${SSG_VERSION_NO_V}.zip" && \
    mkdir -p /usr/share/xml/scap/ssg/content && \
    if [ -f "/ssg.zip" ]; then \
        unzip -jo /ssg.zip "scap-security-guide-${SSG_VERSION_NO_V}/*" -d /usr/share/xml/scap/ssg/content/ && \
        rm -f /ssg.zip; \
    else \
        echo "❌ Failed to download SCAP Security Guide"; exit 1; \
    fi && \
    SCAP_GUIDE=$(find /usr/share/xml/scap/ssg/content -name "ssg-ubuntu*-ds.xml" | sort | tail -n1) && \
    echo "📘 Found SCAP guide: $SCAP_GUIDE" && \
    oscap xccdf eval \
        --remediate \
        --profile xccdf_org.ssgproject.content_profile_cis_level2_server \
        --results /root/oscap-results.xml \
        --report /root/oscap-report.html \
        "$SCAP_GUIDE" || true && \
    rm -rf /usr/share/xml/scap/ssg/content && \
    apt-get remove -y openscap-scanner libopenscap25t64 && \
    apt-get autoremove -y && \
    apt-get clean

# --- 7. Install Python 3.10 (required for Triton Python API compatibility) ---
# Ubuntu 24.04 comes with Python 3.12, but Triton's Python bindings are for Python 3.10
# We need to add the deadsnakes PPA to get Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 7a. Install Ray (full installation with all extras) with Python 3.10 ---
# Note: Using Python 3.10 to match Triton's Python version for Python API compatibility
# CUDA toolkit will be provided via CVMFS (Digital Research Alliance of Canada)
# Install Ray in a virtual environment to avoid PEP 668 and Debian package conflicts
# ray[all] includes: default, serve, tune, rllib, data, train, air, gpu, and more
# tritonclient[all]: Client libraries for connecting to Triton servers (HTTP, gRPC)
RUN python3.10 -m venv /opt/ray && \
    /opt/ray/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/ray/bin/pip install --no-cache-dir "ray[all]" && \
    /opt/ray/bin/pip install --no-cache-dir "tritonclient[all]" && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/profile.d/ray.sh && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/bash.bashrc

# --- 7b. Copy Triton Inference Server from NVIDIA image ---
# Copy Triton server binary, libraries, and Python bindings (Python 3.10 compatible)
# This enables 'import tritonserver' to work in Ray's Python 3.10 environment

# First, copy the entire Triton installation
COPY --from=triton-source /opt/tritonserver /opt/tritonserver

# Copy prepared Python bindings from source stage
COPY --from=triton-source /tmp/triton-bindings/ /tmp/triton-python-bindings/

# Install Python bindings into Ray's Python 3.10 environment
RUN mkdir -p /opt/ray/lib/python3.10/site-packages && \
    # Copy Python bindings from prepared location
    if [ -d "/tmp/triton-python-bindings" ] && [ "$(ls -A /tmp/triton-python-bindings 2>/dev/null)" ]; then \
        cp -r /tmp/triton-python-bindings/* /opt/ray/lib/python3.10/site-packages/ 2>/dev/null || true; \
        rm -rf /tmp/triton-python-bindings; \
        echo "✅ Copied Triton Python bindings to Ray's site-packages"; \
    else \
        echo "⚠️  Triton Python bindings not found in source image - will use binary mode"; \
    fi && \
    # Also check for tritonserver Python modules in /opt/tritonserver/python (from copied binary)
    if [ -d "/opt/tritonserver/python" ]; then \
        cp -r /opt/tritonserver/python/* /opt/ray/lib/python3.10/site-packages/ 2>/dev/null || true; \
    fi && \
    # Set up Python path to include Triton's Python modules
    echo 'export PYTHONPATH="/opt/tritonserver/python:${PYTHONPATH}"' >> /etc/profile.d/ray.sh && \
    echo 'export PYTHONPATH="/opt/tritonserver/python:${PYTHONPATH}"' >> /etc/bash.bashrc

# Set up Triton environment variables
ENV PATH="/opt/tritonserver/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/tritonserver/lib:${LD_LIBRARY_PATH:-}"
ENV PYTHONPATH="/opt/tritonserver/python:${PYTHONPATH:-}"

# Verify Triton installation
RUN echo "🔍 Verifying Triton installation..." && \
    /opt/tritonserver/bin/tritonserver --version && \
    echo "✅ Triton binary available" && \
    (/opt/ray/bin/python -c "import tritonserver; print('✅ Triton Python API available'); print(f'   Location: {tritonserver.__file__}')" 2>&1 || \
     echo "⚠️  Triton Python API not directly importable - will use binary mode or HTTP client")

# Add Ray venv to PATH for all sessions
ENV PATH="/opt/ray/bin:$PATH"

# --- 8. Install NVIDIA Driver if enabled (requires kernel installation) ---
# Note: CUDA toolkit not installed here - will be available via CVMFS
# Build tools are already installed in Step 5 if NVIDIA_INSTALL_ENABLED=true
RUN if [ "$NVIDIA_INSTALL_ENABLED" = "true" ] && [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        mkdir -p /build && cd /build && \
        echo "📥 Downloading NVIDIA driver from ${NVIDIA_DRIVER_URL}..." && \
        wget -q "${NVIDIA_DRIVER_URL}" -O /tmp/NVIDIA.run && \
        echo "📦 Extracting driver..." && \
        chmod +x /tmp/NVIDIA.run && \
        /tmp/NVIDIA.run --extract-only --target /build/nvidia && \
        cd /build/nvidia && \
        ./nvidia-installer --accept-license \
                          --no-questions \
                          --silent \
                          --no-backup \
                          --no-x-check \
                          --no-nouveau-check \
                          --no-systemd \
                          --no-check-for-alternate-installs \
                          --kernel-name=${KERNEL_VERSION} \
                          --kernel-source-path=/lib/modules/${KERNEL_VERSION}/build \
                          --x-prefix=/usr \
                          --x-module-path=/usr/lib/xorg/modules \
                          --x-library-path=/usr/lib && \
        mkdir -p /etc/modules-load.d/ && \
        echo "nvidia" > /etc/modules-load.d/nvidia.conf && \
        echo "nvidia_uvm" >> /etc/modules-load.d/nvidia.conf && \
        echo "nvidia_drm" >> /etc/modules-load.d/nvidia.conf && \
        echo "nvidia_modeset" >> /etc/modules-load.d/nvidia.conf && \
        mkdir -p /dev/nvidia && \
        [ -e /dev/nvidia0 ] || mknod -m 666 /dev/nvidia0 c 195 0 && \
        [ -e /dev/nvidiactl ] || mknod -m 666 /dev/nvidiactl c 195 255 && \
        [ -e /dev/nvidia-uvm ] || mknod -m 666 /dev/nvidia-uvm c 243 0 && \
        [ -e /dev/nvidia-uvm-tools ] || mknod -m 666 /dev/nvidia-uvm-tools c 243 1 ; \
    fi

# --- 9. Configure Autologin based on DISABLE_AUTOLOGIN ---
RUN if [ "$DISABLE_AUTOLOGIN" != "true" ]; then \
        mkdir -p /etc/systemd/system/getty@tty1.service.d && \
        echo '[Service]' > /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=' >> /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM' >> /etc/systemd/system/getty@tty1.service.d/override.conf; \
    else \
        rm -rf /etc/systemd/system/getty@tty1.service.d; \
    fi

# --- 10. Configure Firstboot Service ---
COPY firstboot.service /etc/systemd/system/
COPY firstboot.sh /usr/local/sbin/
RUN if [ "$FIRSTBOOT_ENABLED" = "true" ]; then \
        chmod +x /usr/local/sbin/firstboot.sh && \
        systemctl enable firstboot.service; \
    else \
        rm -f /etc/systemd/system/multi-user.target.wants/firstboot.service && \
        rm -f /usr/local/sbin/firstboot.sh; \
    fi

# --- 11. Enable Core Services ---
RUN systemctl enable \
    rsyslog.service \
    ssh.service

# --- 12. Generate Initramfs for Selected Kernel (if kernel is installed) ---
RUN if [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        update-initramfs -u -k "$KERNEL_VERSION"; \
    fi

# --- 13. Create Ray Directories ---
RUN mkdir -p /etc/ray && \
    mkdir -p /var/log/ray && \
    mkdir -p /tmp/ray && \
    mkdir -p /var/lib/ray

# --- 14. Final Cleanup ---
# Only remove build/dev packages that are actually installed and safe to remove
# Keep essential runtime packages like multipath-tools, software-properties-common, etc.
RUN apt-mark manual libvulkan1 mesa-vulkan-drivers libglvnd0 && \
    for pkg in mesa-common-dev xserver-xorg-dev xorg-dev \
        libx11-dev libxext-dev libxft-dev libxau-dev libxdmcp-dev \
        libxcb1-dev libxcomposite-dev libxcursor-dev libxdamage-dev \
        libxfixes-dev libxfont-dev libxi-dev libxinerama-dev \
        libxkbfile-dev libxmu-dev libxpm-dev libxrandr-dev \
        libxrender-dev libxres-dev libxss-dev libxt-dev libxtst-dev \
        libxv-dev libxvmc-dev libxxf86dga-dev libxxf86vm-dev \
        libgl-dev libglvnd-dev libglvnd-core-dev libglx-dev \
        libegl-dev libgles-dev libdmx-dev libfontconfig-dev \
        build-essential dkms gcc g++ make pkg-config dpkg-dev \
        libfreetype-dev libpng-dev uuid-dev libexpat1-dev \
        libpython3-dev libpython3.10-dev python3-dev python3.10-dev \
        python-babel-localedata humanity-icon-theme iso-codes; do \
        dpkg -l "$pkg" >/dev/null 2>&1 && apt-get purge -y "$pkg" || true; \
    done && \
    # Only remove initramfs-tools if kernel install was disabled
    if [ "$KERNEL_INSTALL_ENABLED" != "true" ]; then \
        apt-get purge -y initramfs-tools || true; \
    fi && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf \
        /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /build \
        /var/log/apt/* \
        /usr/share/doc \
        /usr/share/man \
        /usr/share/locale \
        /usr/share/locale-langpack \
        /usr/share/info \
        /NVIDI* \
        /root/.cache \
        /root/.wget-hsts && \
    find / -name '*.bash_history' -delete && \
    find /var/log/ -type f -exec rm -f {} + && \
    find / -name '.wget-hsts' -delete && \
    find / -name '.cache' -exec rm -rf {} +

# --- 15. Unmask and Enable Services ---
RUN systemctl unmask \
    systemd-udevd.service \
    systemd-udevd-kernel.socket \
    systemd-udevd-control.socket \
    systemd-modules-load.service \
    sys-kernel-config.mount \
    sys-kernel-debug.mount \
    sys-fs-fuse-connections.mount \
    systemd-remount-fs.service \
    getty.target \
    systemd-logind.service \
    systemd-vconsole-setup.service \
    systemd-timesyncd.service && \
    systemctl enable \
    systemd-udevd.service \
    systemd-modules-load.service \
    getty@tty1.service \
    systemd-logind.service \
    ssh.service \
    rsyslog.service

# --- 16. Systemd-compatible boot (Warewulf) ---
#STOPSIGNAL SIGRTMIN+3
#CMD ["/sbin/init"]
