#!/bin/bash
# Auto-activate conda environment based on directory
# Add this to your ~/.bashrc:
# source /p/rlprojects/RND/auto_activate_env.sh

# Directory to environment mapping
# Format: "directory_path:environment_name"
declare -A DIR_ENV_MAP=(
    ["/p/rlprojects/RND"]="exploration"
    ["/p/rlprojects/202601ODI"]="robust"
)

# Store the original cd command
if [ -z "$ORIGINAL_CD" ]; then
    export ORIGINAL_CD=$(which cd)
fi

# Function to initialize conda if needed
init_conda() {
    if command -v conda &> /dev/null; then
        # If conda is not initialized (CONDA_SHLVL not set or conda activate doesn't work)
        if [ -z "$CONDA_SHLVL" ] || ! conda activate --help &>/dev/null 2>&1; then
            # Try to find and source conda.sh
            local conda_base
            # First try to get it from conda command
            conda_base=$(conda info --base 2>/dev/null)
            # If that fails, try common locations
            if [ -z "$conda_base" ] || [ ! -f "${conda_base}/etc/profile.d/conda.sh" ]; then
                # Try to find conda in common locations
                if [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
                    conda_base="$HOME/.conda"
                elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
                    conda_base="/opt/conda"
                fi
            fi
            # Source conda.sh if found
            if [ -n "$conda_base" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
                source "${conda_base}/etc/profile.d/conda.sh" 2>/dev/null
            fi
        fi
    fi
}

# Function to check and activate conda environment
check_and_activate_env() {
    local current_dir=$(pwd)
    local target_env=""
    local target_dir=""
    
    # Check which directory we're in (most specific match first)
    for dir in "${!DIR_ENV_MAP[@]}"; do
        if [[ "$current_dir" == "$dir"* ]]; then
            # Use the longest matching directory path
            if [ -z "$target_dir" ] || [ ${#dir} -gt ${#target_dir} ]; then
                target_dir="$dir"
                target_env="${DIR_ENV_MAP[$dir]}"
            fi
        fi
    done
    
    # Initialize conda if needed
    if command -v conda &> /dev/null; then
        init_conda
        
        if [ -n "$target_env" ]; then
            # We're in a mapped directory - activate the corresponding environment
            if [ "$CONDA_DEFAULT_ENV" != "$target_env" ]; then
                # Check if environment exists (handle both exact match and with path)
                if conda env list 2>/dev/null | grep -qE "^${target_env}[[:space:]]"; then
                    # Deactivate current environment if different
                    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "$target_env" ]; then
                        # Only deactivate if we're switching to a different environment
                        # Deactivate all nested environments first
                        while [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; do
                            conda deactivate 2>/dev/null || break
                            # Small delay to let deactivation complete
                            sleep 0.05 2>/dev/null || true
                        done
                    fi
                    # Activate the target environment
                    conda activate "$target_env" 2>/dev/null
                fi
            fi
        else
            # We're not in any mapped directory - deactivate if a mapped env is active
            for env in "${DIR_ENV_MAP[@]}"; do
                if [ "$CONDA_DEFAULT_ENV" == "$env" ]; then
                    conda deactivate 2>/dev/null
                    break
                fi
            done
        fi
    fi
}

# Override cd command
cd() {
    builtin cd "$@"
    check_and_activate_env
}

# Also check on prompt (in case directory changes via other means)
if [ -z "$RND_AUTO_ACTIVATE_LOADED" ]; then
    export RND_AUTO_ACTIVATE_LOADED=1
    # Check immediately if we're already in RND directory
    check_and_activate_env
fi
