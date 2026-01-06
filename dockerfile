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

# --- 4. Install Core Tools, Debugging, and Dependencies ---
RUN apt-get update && apt-get install -y \
    sudo \
    openssh-server \
    openssh-client \
    net-tools \
    iproute2 \
    pciutils \
    lvm2 \
    nfs-common \
    multipath-tools \
    ifupdown \
    rsync \
    curl \
    wget \
    vim \
    less \
    htop \
    sysstat \
    cron \
    ipmitool \
    smartmontools \
    lm-sensors \
    netplan.io \
    unzip \
    gnupg \
    ansible \
    systemd \
    systemd-sysv \
    dbus \
    initramfs-tools \
    socat \
    conntrack \
    ebtables \
    ethtool \
    ipset \
    iptables \
    tcpdump \
    strace \
    lsof \
    jq \
    git \
    iputils-ping \
    lsb-release \
    bash-completion \
    cgroup-tools \
    auditd \
    apt-transport-https \
    software-properties-common \
    ca-certificates \
    kmod \
    numactl \
    apt-utils \
    netbase \
    libnuma1 \
    libpam0g \
    libyaml-0-2 \
    libjson-c5 \
    libssl3 \
    libcurl4 \
    libdbus-1-3 \
    rsyslog \
    logrotate \
    python3 \
    python3-pip \
    python3-venv && \
    if [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        apt-get install -y \
            linux-image-${KERNEL_VERSION} \
            linux-headers-${KERNEL_VERSION} \
            linux-modules-${KERNEL_VERSION} \
            linux-modules-extra-${KERNEL_VERSION} && \
        ln -s /usr/src/linux-headers-${KERNEL_VERSION} /lib/modules/${KERNEL_VERSION}/build; \
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

# --- 5. Install Ray (latest version with GPU support) ---
# Note: CUDA toolkit will be provided via CVMFS (Digital Research Alliance of Canada)
# Install Ray in a virtual environment to avoid PEP 668 and Debian package conflicts
RUN python3 -m venv /opt/ray && \
    /opt/ray/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/ray/bin/pip install --no-cache-dir "ray[default,gpu]" && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/profile.d/ray.sh && \
    echo 'export PATH="/opt/ray/bin:$PATH"' >> /etc/bash.bashrc

# Add Ray venv to PATH for all sessions
ENV PATH="/opt/ray/bin:$PATH"

# --- 6. Install NVIDIA Driver if enabled (requires kernel installation) ---
# Note: CUDA toolkit not installed here - will be available via CVMFS
RUN if [ "$NVIDIA_INSTALL_ENABLED" = "true" ] && [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        apt-get update && apt-get install -y \
            build-essential \
            pkg-config \
            xorg-dev \
            libx11-dev \
            libxext-dev \
            libglvnd-dev && \
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

# --- 7. Configure Autologin based on DISABLE_AUTOLOGIN ---
RUN if [ "$DISABLE_AUTOLOGIN" != "true" ]; then \
        mkdir -p /etc/systemd/system/getty@tty1.service.d && \
        echo '[Service]' > /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=' >> /etc/systemd/system/getty@tty1.service.d/override.conf && \
        echo 'ExecStart=-/sbin/agetty --autologin root --noclear %I $TERM' >> /etc/systemd/system/getty@tty1.service.d/override.conf; \
    else \
        rm -rf /etc/systemd/system/getty@tty1.service.d; \
    fi

# --- 8. Configure Firstboot Service ---
COPY firstboot.service /etc/systemd/system/
COPY firstboot.sh /usr/local/sbin/
RUN if [ "$FIRSTBOOT_ENABLED" = "true" ]; then \
        chmod +x /usr/local/sbin/firstboot.sh && \
        systemctl enable firstboot.service; \
    else \
        rm -f /etc/systemd/system/multi-user.target.wants/firstboot.service && \
        rm -f /usr/local/sbin/firstboot.sh; \
    fi

# --- 9. Enable Core Services ---
RUN systemctl enable \
    rsyslog.service \
    ssh.service \
    auditd.service

# --- 10. Generate Initramfs for Selected Kernel (if kernel is installed) ---
RUN if [ "$KERNEL_INSTALL_ENABLED" = "true" ]; then \
        update-initramfs -u -k "$KERNEL_VERSION"; \
    fi

# --- 11. Create Ray Directories ---
RUN mkdir -p /etc/ray && \
    mkdir -p /var/log/ray && \
    mkdir -p /tmp/ray && \
    mkdir -p /var/lib/ray

# --- 12. Final Cleanup ---
RUN apt-mark manual libvulkan1 mesa-vulkan-drivers libglvnd0 && \
    apt-get purge -y \
        mesa-common-dev xserver-xorg-dev xorg-dev \
        libx*dev libgl*dev libegl*dev libgles*dev \
        libx11-dev libxext-dev libxft-dev \
        build-essential dkms gcc make pkg-config \
        libfreetype-dev libpng-dev uuid-dev libexpat1-dev \
        python-babel-localedata \
        humanity-icon-theme \
        iso-codes \
        xorg-dev \
        libx11-dev \
        libxext-dev \
        initramfs-tools && \
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

# --- 13. Unmask and Enable Services ---
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

# --- 14. Systemd-compatible boot (Warewulf) ---
#STOPSIGNAL SIGRTMIN+3
#CMD ["/sbin/init"]
