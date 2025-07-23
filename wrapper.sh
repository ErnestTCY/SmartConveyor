#!/bin/bash

cd ~/Desktop/fyp1/ || exit 1

# Launch each script in its own gnome-terminal and get the PID of that terminal
gnome-terminal -- bash -c "python3 cleanup.py; exec bash" &
PID1=$!
gnome-terminal -- bash -c "python3 anomaly_detection-v1.py; exec bash" &
PID2=$!
gnome-terminal -- bash -c "python3 robotic_arm_control.py; exec bash" &
PID3=$!

echo "Scripts launched in separate terminals."
echo "Press 'q' then Enter to terminate all terminals..."

# Wait for 'q' keypress
while true; do
    read -r -n1 key
    if [[ $key == "q" ]]; then
        echo -e "\nKilling terminals..."
        kill $PID1 $PID2 $PID3 2>/dev/null
        break
    fi
done

echo "All terminals terminated."
