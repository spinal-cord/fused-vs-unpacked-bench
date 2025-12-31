#!/bin/bash

# Create output directory for benchmark results
OUTPUT_DIR="benchmark_results"
mkdir -p "$OUTPUT_DIR"

# Function to check if benchmark file exists and is executable
file_check() {
    if [[ ! -f "./benchmark" ]]; then
        echo "Error: Benchmark file './benchmark' not found."
        exit 1
    fi

    if [[ ! -x "./benchmark" ]]; then
        echo "Error: Benchmark file './benchmark' is not executable."
        exit 1
    fi
}

# Function to get CPU information based on OS
get_cpu_info() {
    local os_type="$1"
    
    if [[ "$os_type" == "Darwin" ]]; then
        # macOS
        cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    else
        # Linux
        # Try multiple methods to get CPU info
        if [[ -f /proc/cpuinfo ]]; then
            cpu_info=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' 2>/dev/null || echo "Unknown")
        fi
        
        # If still unknown, try lscpu
        if [[ -z "$cpu_info" || "$cpu_info" == "Unknown" ]]; then
            cpu_info=$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' || echo "Unknown")
        fi
    fi
    
    # If still unknown, use architecture
    if [[ -z "$cpu_info" || "$cpu_info" == "Unknown" ]]; then
        arch=$(uname -m)
        if [[ "$arch" == "arm64" || "$arch" == "aarch64" ]]; then
            cpu_info="ARM64"
        elif [[ "$arch" == "x86_64" ]]; then
            cpu_info="x86_64"
        else
            cpu_info="$arch"
        fi
    fi
    
    echo "$cpu_info"
}

# Function to get OS version information
get_os_info() {
    local os_type="$1"
    
    if [[ "$os_type" == "Darwin" ]]; then
        # macOS
        os_major=$(sw_vers -productVersion | cut -d. -f1)
        os_name="macOS${os_major}"
    else
        # Linux - try to get distribution info
        if [[ -f /etc/os-release ]]; then
            # Read the os-release file
            source /etc/os-release
            if [[ -n "$PRETTY_NAME" ]]; then
                # Clean up the distribution name for filename
                os_name=$(echo "$PRETTY_NAME" | sed 's/[^a-zA-Z0-9._-]/_/g')
            elif [[ -n "$NAME" && -n "$VERSION_ID" ]]; then
                os_name="${NAME}_${VERSION_ID}"
            else
                os_name="Linux_$(uname -r)"
            fi
        else
            # Fallback to kernel version
            os_name="Linux_$(uname -r | sed 's/[^a-zA-Z0-9._-]/_/g')"
        fi
    fi
    
    echo "$os_name"
}

# Function to get UTC timestamp for filename (safe characters only)
get_utc_timestamp_filename() {
    # Format for filename: YYYY-MM-DD-HH-MM-UTC
    # Using only safe characters (no + or other special chars that might cause issues)
    date -u +"%Y-%m-%d-%H-%M-UTC"
}

# Function to get UTC date for display (with +0000)
get_utc_date_display() {
    # Format for display: YYYY-MM-DD HH:MM:SS UTC+0000
    date -u +"%Y-%m-%d %H:%M:%S UTC+0000"
}

# Build the benchmark
echo "Building the benchmark..."
make

# Check if benchmark file exists and is executable
file_check

# Remove macOS quarantine attribute if on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Removing macOS quarantine attribute for the file \"benchmark\""
    xattr -d com.apple.quarantine ./benchmark 2>/dev/null || true
fi

# Get system information
os_type=$(uname)
cpu_info=$(get_cpu_info "$os_type")
os_info=$(get_os_info "$os_type")
utc_timestamp=$(get_utc_timestamp_filename)
utc_date_display=$(get_utc_date_display)

# Create filename with cleaned up system info and UTC timestamp
clean_cpu=$(echo "$cpu_info" | tr -cd 'a-zA-Z0-9._-' | sed 's/__*/_/g')
clean_os=$(echo "$os_info" | tr -cd 'a-zA-Z0-9._-' | sed 's/__*/_/g')
filename="${OUTPUT_DIR}/benchmark_${clean_cpu}_${clean_os}_${utc_timestamp}.txt"

# Ensure the output directory exists (in case mkdir -p failed earlier)
mkdir -p "$OUTPUT_DIR"

# Execute the benchmark
echo "Running benchmark on ${os_info}..."

# Tee command shows output in terminal AND saves to file
{
    echo "CPU: ${cpu_info}"
    echo "OS: ${os_info}"
    echo "Date (UTC): ${utc_date_display}"
    echo "------------------------"
    ./benchmark
} | tee "$filename"

echo ""
echo "Results saved to: $filename"
echo "All benchmark results are stored in the '${OUTPUT_DIR}/' directory"