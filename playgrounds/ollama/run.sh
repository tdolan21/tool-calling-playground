#!/bin/bash

# Function to kill all child processes
cleanup() {
    echo "Stopping all processes..."
    pkill -P $$
    exit
}

# Set up trap to call cleanup function when script receives SIGINT or SIGTERM
trap cleanup SIGINT SIGTERM

# Start the Python server
python image_gen/server.py &

# Start the Streamlit server
streamlit run playground.py &

# Wait for all background processes to finish
wait
