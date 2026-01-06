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
    # Essential utilities
    curl \
    wget \
    less \
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

# --- 6. Install Ray (full installation with all extras) and Triton support ---
# Note: CUDA toolkit will be provided via CVMFS (Digital Research Alliance of Canada)
# Install Ray in a virtual environment to avoid PEP 668 and Debian package conflicts
# ray[all] includes: default, serve, tune, rllib, data, train, air, gpu, and more
# tritonclient[all]: Client libraries for connecting to Triton servers (HTTP, gRPC)
# nvidia-pytriton: PyTriton wrapper for Triton's Python API (embedded mode with Ray Serve)
RUN python3 -m venv /opt/ray && \
    /opt/ray/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/ray/bin/pip install --no-cache-dir "ray[all]" && \
    /opt/ray/bin/pip install --no-cache-dir "tritonclient[all]" && \
    /opt/ray/bin/pip install --no-cache-dir "nvidia-pytriton" && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/profile.d/ray.sh && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/bash.bashrc

# Add Ray venv to PATH for all sessions
ENV PATH="/opt/ray/bin:$PATH"

# --- 7. Install NVIDIA Driver if enabled (requires kernel installation) ---
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

# --- 8. Configure Autologin based on DISABLE_AUTOLOGIN ---
RUN if [ "$DISABLE_AUTOLOGIN" != "true" ]; then \
        mkdir -p /etc/systemd/system/getty@tty1.service.d && \
        echo '[Service]' > /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=' >> /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM' >> /etc/systemd/system/getty@tty1.service.d/override.conf; \
    else \
        rm -rf /etc/systemd/system/getty@tty1.service.d; \
    fi

# --- 9. Configure Firstboot Service ---
COPY firstboot.service /etc/systemd/system/
COPY firstboot.sh /usr/local/sbin/
RUN if [ "$FIRSTBOOT_ENABLED" = "true" ]; then \
        chmod +x /usr/local/sbin/firstboot.sh && \
        systemctl enable firstboot.service; \
    else \
        rm -f /etc/systemd/system/multi-user.target.wants/firstboot.service && \
        rm -f /usr/local/sbin/firstboot.sh; \
    fi

# --- 10. Enable Core Services ---
RUN systemctl enable \
    rsyslog.service \
    ssh.service \
    auditd.service

# --- 11. Generate Initramfs for Selected Kernel (if kernel is installed) ---
RUN if [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        update-initramfs -u -k "$KERNEL_VERSION"; \
    fi

# --- 12. Create Ray Directories ---
RUN mkdir -p /etc/ray && \
    mkdir -p /var/log/ray && \
    mkdir -p /tmp/ray && \
    mkdir -p /var/lib/ray

# --- 13. Final Cleanup ---
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
        libpython3-dev libpython3.12-dev python3-dev python3.12-dev \
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

# --- 14. Unmask and Enable Services ---
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
    rsyslog.service \
    auditd.service

# --- 15. Systemd-compatible boot (Warewulf) ---
#STOPSIGNAL SIGRTMIN+3
#CMD ["/sbin/init"]
