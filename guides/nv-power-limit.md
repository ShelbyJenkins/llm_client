
# Limiting power in Nvidia GPUs 

See this post for more information: https://www.pugetsystems.com/labs/hpc/quad-rtx3090-gpu-power-limiting-with-systemd-and-nvidia-smi-1983/

## Creating the bash script

### 1. Open a terminal and switch to the root user using the following command:

`sudo -i`

### 2. Create a new file named nv-power-limit.sh in the /usr/local/sbin/ directory using a text editor (e.g., nano):

`nano /usr/local/sbin/nv-power-limit.sh`

### 3. Copy and paste the script content into the file: 

```bash
#!/usr/bin/env bash

# Set power limits on all NVIDIA GPUs
# See shelbyjenkins.github for more details
# Make sure nvidia-smi exists 
command -v nvidia-smi &> /dev/null || { echo >&2 "nvidia-smi not found ... exiting."; exit 1; }

# Define GPU enabled states (1 for enabled, 0 for disabled)
declare -A gpu_enabled=(
    [0]=1
    [1]=1
    [2]=0
    [3]=0
    [4]=0
    [5]=0
)

# Define desired power limits for each GPU (in Watts)
declare -A gpu_power_limits=(
    [0]=222
    [1]=222
    [2]=0
    [3]=0
    [4]=0
    [5]=0
)

# Function to set power limit
set_power_limit() {
    local gpu_id=$1
    local limit=$2
    /usr/bin/nvidia-smi -i $gpu_id --persistence-mode=1
    /usr/bin/nvidia-smi -i $gpu_id --power-limit=$limit
}
for gpu_id in "${!gpu_enabled[@]}"; do
    if [[ ${gpu_enabled[$gpu_id]} -eq 1 ]]; then
        # Fetch the maximum power limit for the current GPU
        max_power_limit=$(nvidia-smi -i $gpu_id -q -d POWER | grep 'Max Power Limit' | awk '{print $5}' | grep -oE '[0-9]+([.][0-9]+)?')
        if [[ -z "$max_power_limit" || "$max_power_limit" == "N/A" ]]; then
            echo "GPU $gpu_id: Max Power Limit not available."
            continue
        fi

        # Check if the desired limit is less than the max
        if [[ ${gpu_power_limits[$gpu_id]} -le $(printf "%.0f" "$max_power_limit") ]]; then
            echo "Setting power limit for GPU $gpu_id to ${gpu_power_limits[$gpu_id]} Watts."
            set_power_limit $gpu_id ${gpu_power_limits[$gpu_id]}
        else
            echo "FAIL! Desired power limit for GPU $gpu_id is above the max allowable limit of $max_power_limit Watts."
        fi
    else
        echo "GPU $gpu_id is disabled."
    fi
done

exit 0
```
Save the file and exit the text editor (in nano, press Ctrl+X, then Y, and finally Enter).

### 4. Set the file permissions so only root can edit and excute:

`chmod 744 /usr/local/sbin/nv-power-limit.sh`


## Setup systemd to run nv-power-limit.service at boot

### 1. Create the /usr/local/etc/systemd directory if it doesn't exist:

`mkdir -p /usr/local/etc/systemd`

### 2. Create a new file named nv-power-limit.service in the /usr/local/etc/systemd directory using a text editor (e.g., nano):

`nano /usr/local/etc/systemd/nv-power-limit.service`

### 3. Copy and paste the following content into the file:

```
# Set power limits on all NVIDIA GPUs
# See shelbyjenkins.github for more details
[Unit]
Description=NVIDIA GPU Set Power Limit
After=syslog.target systemd-modules-load.service
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
User=root
Environment="PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
ExecStart=/usr/local/sbin/nv-power-limit.sh

[Install]
WantedBy=multi-user.target
```
Save the file and exit the text editor (in nano, press Ctrl+X, then Y, and finally Enter).

### 4. Set the file permissions so only root can edit and excute:

`chmod 644 /usr/local/etc/systemd/nv-power-limit.service`

### 5. Create a symbolic link to the unit file in the /etc/systemd/system directory: 

`ln -s /usr/local/etc/systemd/nv-power-limit.service /etc/systemd/system/nv-power-limit.service`

### 6. Verify that the link was created correctly by listing the contents of the /etc/systemd/system directory:

`ls -l /etc/systemd/system`

look for something like:

`lrwxrwxrwx 1 root root   45 Apr  7 15:28  nv-power-limit.service -> /usr/local/etc/systemd/nv-power-limit.service`

### 8. Start the nv-power-limit.service to test if it's working properly:

`systemctl start nv-power-limit.service`

### 9. Check the status of the service: 

`systemctl status nv-power-limit.service`

### 10. Enable the service to start at boot time: 

`systemctl enable nv-power-limit.service`

### 11. Reboot

`reboot`

### 12. After the system restarts, verify that the GPU power limit is set correctly by running: 

`nvidia-smi -q -d POWER`

## Making Changes

### 1. Open, edit, and save the script

`nano /usr/local/sbin/nv-power-limit.sh`

### 2. Reload systemd

`sudo systemctl daemon-reload`

### 3. Restart the service

`sudo systemctl restart nv-power-limit.service`

### 4. Check the service status

`sudo systemctl status nv-power-limit.service`