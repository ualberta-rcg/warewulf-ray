#!/bin/bash
# Exit with error code if any command fails
set -e
# Log everything we do
exec 1> >(logger -s -t $(basename $0)) 2>&1

echo "Starting first boot configuration..."
export DEBIAN_FRONTEND=noninteractive

# Setup System
EXIT_CODE=0
for playbook in $(ls /etc/ansible/playbooks/*.yaml | sort); do
    echo "Running playbook: $playbook"
    if ! ansible-playbook "$playbook"; then
        echo "ERROR: Playbook $playbook failed!"
        EXIT_CODE=1
        break
    fi
done

if [ $EXIT_CODE -eq 0 ]; then
    echo "First boot configuration completed successfully"
    rm /etc/systemd/system/firstboot.service
    #rm -- "$0"
    systemctl daemon-reload
    echo "First boot Service Removed"
else
    echo "First boot configuration failed with errors"
fi

# Return proper exit code
exit $EXIT_CODE
