#!/bin/bash

# Script to open Activity Monitor with the GPU tab active
# Only works on macOS

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script only works on macOS"
    exit 1
fi

# Check if Activity Monitor is already running
if pgrep "Activity Monitor" > /dev/null; then
    echo "Closing existing Activity Monitor"
    osascript -e 'tell application "Activity Monitor" to quit'
    sleep 1
fi

# Open Activity Monitor and switch to GPU tab
echo "Opening Activity Monitor with GPU tab"
open -a "Activity Monitor"

# Use AppleScript to switch to the GPU tab
sleep 1
osascript <<EOF
tell application "System Events"
    tell process "Activity Monitor"
        tell radio button "GPU" of radio group 1 of group 2 of toolbar 1 of window 1
            click
        end tell
    end tell
end tell
EOF

echo "Activity Monitor opened with GPU tab. Keep this window visible while running your script." 