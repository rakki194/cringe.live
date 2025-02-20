---
title: "Tracking Training Changes"
description: "Guide to keeping track of changes during LoRA training using automated scripts"
summary: "Learn how to use automated scripts to organize and track your LoRA training process, including managing model versions, backing up configurations, and maintaining clean training workspaces."
weight: 8
bookToC: false
bookFlatSection: false
aliases:
  - /en/docs/yiff_toolkit/lora_training/Tracking-Training-Changes/
  - /en/docs/yiff_toolkit/lora_training/Tracking-Training-Changes
  - /en/docs/yiff_toolkit/lora_training/Tracking_Training_Changes/
  - /en/docs/yiff_toolkit/lora_training/Tracking_Training_Changes
  - "/en/docs/yiff_toolkit/lora_training/Tracking Training Changes/"
  - "/en/docs/yiff_toolkit/lora_training/Tracking Training Changes"
  - /docs/yiff_toolkit/lora_training/Tracking-Training-Changes/
  - /docs/yiff_toolkit/lora_training/Tracking-Training-Changes
  - /docs/yiff_toolkit/lora_training/Tracking_Training_Changes/
  - /docs/yiff_toolkit/lora_training/Tracking_Training_Changes
  - "/docs/yiff_toolkit/lora_training/Tracking Training Changes/"
  - "/docs/yiff_toolkit/lora_training/Tracking Training Changes"
---

## Overview

When training LoRAs, it's crucial to keep track of your training configurations, model versions, and changes. This guide explains how to use automated scripts to track changes during training.

## The Training Script

The training script provided below handles several important aspects of tracking. It maintains a record of your Git repository state to track code changes and versions over time. The script also preserves your training configurations by creating backups of key files, ensuring you can reference and reproduce your training settings later. Additionally, it saves sample prompts that were used during training for future reference. To keep your workspace organized, the script automatically cleans up any failed training runs by removing their output directories.

Here's how to use it:

```zsh
#!/usr/bin/env zsh
set -e -o pipefail

NAME=foobar
# Optional variables
TRAINING_DIR="/home/kade/datasets/foobar"
# STEPS=
# OUTPUT_DIR=

SD_SCRIPT="${SD_SCRIPT:-sdxl_train_network.py}"
SD_REPO="${SD_REPO:-$HOME/source/repos/sd-scripts}"

# alpha=1 @ dim=16 is the same lr than alpha=4 @ dim=256
args=(
    # Training settings here
)

# ===== Environment Setup =====
source "$HOME/toolkit/zsh/train_functions.zsh"
# Setup variables and training arguments
setup_training_vars "$NAME"
args+=(
    # Add the output and dataset arguments
    --output_dir="$OUTPUT_DIR/$NAME"
    --output_name="$NAME"
    --log_prefix="$NAME-"
    --logging_dir="$OUTPUT_DIR/logs"

    --max_train_steps=$STEPS
    --dataset_config="$TRAINING_DIR/config.toml"
    #--train_data_dir="$TRAINING_DIR"
    --sample_prompts="$TRAINING_DIR/sample-prompts.txt"
    # Script arguments
    "$@"
)

LYCORIS_REPO=$(get_lycoris_repo)

# Set cleanup trap for both error and normal exit
trap cleanup_empty_output EXIT TERM
# Copies the script itself and repositories' commits hashes to the output directory
store_commits_hashes "$SD_REPO" "$LYCORIS_REPO"

# ===== Run Training Script =====
run_training_script "$SD_REPO/$SD_SCRIPT" "${args[@]}"
```

## train_functions.zsh

Create a file at `$HOME/toolkit/zsh/train_functions.zsh` with the following content. This file contains helper functions used by the training script for tasks such as setting up training variables, running the training script, storing Git commit hashes, and cleaning up output directories.

If you need to use a Conda environment, note that the helper function `setup_conda_env` is provided. However, since Conda activation is no longer integrated into the main training script, you should activate your Conda environment manually or call `setup_conda_env` before running the training script.

```zsh
#!/bin/zsh

# Functions for sd-scripts training scripts
# Executes a training script located at the specified path with the provided arguments.
#
# Parameters:
# - script_path: The path to the training script to be executed.
# - args_array: An array of arguments to be passed to the training script.
#
# Behavior:
# - Changes the current directory to the directory of the script.
# - If the DEBUG environment variable is set, it prints the working directory and the arguments.
# - Executes the script using `python` and captures the exit code.
# - Returns to the original directory before exiting.
#
# Returns:
# - The exit code of the executed script.
run_training_script() {
    local script_path="$1"
    local args_array=(${@:2}) # Get all arguments after the first one

    # Store the current directory
    local current_dir=$(pwd)

    # Change to script directory
    local script_dir=$(dirname "$script_path")
    local script_name=$(basename "$script_path")
    cd "$script_dir" || return 1

    # Test if the script exists
    [[ ! -f "$script_name" ]] && echo "\e[31mERROR\e[0m: Script not found: $script_name" && return 1

    echo "Working directory: $(pwd)\nRunning $script_name arguments:"
    for arg in "${args_array[@]}"; do
        echo "  $arg"
    done

    if [[ -n "$DEBUG" ]]; then
        echo "This was a dry run, exiting." | tee "$OUTPUT_DIR/$NAME/sdscripts.log"
        local exit_code=0
    else
        python "$(basename "$script_path")" "${args_array[@]}" | tee "$OUTPUT_DIR/$NAME/sdscripts.log"
        local exit_code=$?
    fi

    # Return to original directory
    cd "$current_dir"

    return $exit_code
}

# Sets up default variables for training
#
# Parameters:
# - name: The name of the training run
#
# Returns:
# - Sets the following global variables:
#   - DATASET_NAME: Base name without version/steps suffix
#   - TRAINING_DIR: Directory containing training data
#   - STEPS: Number of training steps
#   - OUTPUT_DIR: Base output directory
setup_training_vars() {
    local name="$1"

    # Declare globals that will be used by the main script
    typeset -g DATASET_NAME="${name%-*}"
    typeset -g TRAINING_DIR="${TRAINING_DIR:-"${HOME}/datasets/${DATASET_NAME}"}"
    typeset -g STEPS=${STEPS:-"${name##*[^0-9]}"}
    typeset -g OUTPUT_DIR="${HOME}/output_dir"

    echo "\e[35moutput_name\e[0m: $name, \e[35msteps\e[0m: $STEPS, \e[35mtraining_dir\e[0m: $(realpath --relative-to=. $TRAINING_DIR), \e[35moutput_dir\e[0m: $(realpath --relative-to=. \"$OUTPUT_DIR/$name\")"
    
    # ===== Validation =====
    [[ ! -d "$TRAINING_DIR" ]] && echo "ERROR: Training directory not found" && exit 1
    if [[ -d "$OUTPUT_DIR/$name" ]]; then
        echo "ERROR: Output directory already exists: $OUTPUT_DIR/$name"
        exit 1
    fi
}

# Sets up and activates a specified Conda environment.
#
# Parameters:
# - env_name: The name of the Conda environment to activate.
# - conda_path: (Optional) The path to the Conda installation. Defaults to $HOME/miniconda3.
#
# Behavior:
# - Checks if the environment name is provided and if the Conda installation exists.
# - Initializes Conda for the current shell session.
# - Activates the specified Conda environment and verifies its activation.
#
# Returns:
# - 0 on success, or 1 if any error occurs.
setup_conda_env() {
    local env_name="$1"
    [[ -z "$env_name" ]] && echo "\e[31mERROR\e[0m: Environment name required" && return 1

    local conda_path="${2:-$HOME/miniconda3}"
    [[ ! -d "$conda_path" ]] && echo "\e[31mERROR\e[0m: Conda installation not found at $conda_path" && return 1

    # Initialize conda for the shell session
    if __conda_setup="$($conda_path/bin/conda 'shell.zsh' 'hook' 2>/dev/null)" && eval "$__conda_setup"; then
        unset __conda_setup
    else
        echo "\e[31mERROR\e[0m: Failed to initialize conda environment" && return 1
    fi

    # Activate conda environment
    conda activate "$env_name"
    if [ $? -ne 0 ]; then
        echo "\e[31mERROR\e[0m: Failed to activate conda environment: $env_name"
        return 1
    fi
    echo "Conda environment: $CONDA_PREFIX"

    # Verify environment activation
    if ! conda env list | grep -q "^${env_name} "; then
        echo "ERROR: Environment $env_name not found"
        return 1
    fi
}

# Stores the commit hashes of specified Git repositories and copies the script to an output directory.
#
# Parameters:
# - output_dir: The directory where the commit hashes will be stored.
# - repo_path: One or more paths to Git repositories.
#
# Behavior:
# - Creates the output directory if it does not exist.
# - Copies the current script to the output directory.
# - Iterates over each repository path, checking if it is a valid Git repository.
# - Retrieves the current commit SHA for each repository and writes it to an output file.
# - Generates a SHA-1 hash of the script and appends it to the output file.
#
# Returns:
# - 0 on success, or 1 if any error occurs.
store_commits_hashes() {
    local output_dir="$OUTPUT_DIR/$NAME"
    local output_file="$output_dir/repos.git"
    [[ ! -d "$output_dir" ]] && mkdir -p "$output_dir"
    : >"$output_file"

    local summary=""
    local res=0

    for repo_path in "$@"; do
        local repo_name=$(basename "$repo_path")
        if [[ -d "$repo_path/.git" ]]; then
            if local commit_sha=$(git -C "$repo_path" rev-parse HEAD 2>/dev/null); then
                if local branch_name=$(git -C "$repo_path" rev-parse --abbrev-ref HEAD 2>/dev/null); then
                    echo "$repo_path: ($branch_name) $commit_sha" >>"$output_file"
                    summary+="✓ $repo_name: $repo_path ${commit_sha:0:8} ($branch_name)\n"
                else
                    echo "$repo_path: $commit_sha (Failed to get branch)" >>"$output_file"
                    summary+="⚠️  $repo_name: $repo_path ${commit_sha:0:8} (Failed to get branch)\n"
                    res=1
                fi
            else
                echo "$repo_path: Git command failed" >>"$output_file"
                summary+="⚠️  $repo_name: $repo_path (Git command failed) \n"
                res=1
            fi
        else
            echo "$repo_path: Not a git repository" >>"$output_file"
            summary+="⚠️  $repo_name: Not a git repository $repo_path\n"
            res=1
        fi
    done

    local script_path=$(readlink -f "$ZSH_SCRIPT")
    cp "$script_path" "$output_dir/$(basename "$script_path")"
    [[ -n "$DEBUG" ]] && echo "Copied $script_path to $output_dir"

    local script_sha=$(sha1sum "$script_path" | cut -f1 -d' ')
    echo "$script_path: $script_sha" >>"$output_file"
    summary+="✓ Training script: $ZSH_SCRIPT ${script_sha:0:8}\n"

    local config_sha=$(sha1sum "$TRAINING_DIR/config.toml" | cut -f1 -d' ')
    local prompts_sha=$(sha1sum "$TRAINING_DIR/sample-prompts.txt" | cut -f1 -d' ')
    cp "$TRAINING_DIR/config.toml" "$output_dir/config.toml"
    cp "$TRAINING_DIR/sample-prompts.txt" "$output_dir/sample-prompts.txt"
    echo "$TRAINING_DIR/config.toml: $config_sha" >>"$output_file"
    echo "$TRAINING_DIR/sample-prompts.txt: $prompts_sha" >>"$output_file"
    summary+="✓ Training config: $TRAINING_DIR/config.toml ${config_sha:0:8}\n"
    summary+="✓ Training prompts: $TRAINING_DIR/sample-prompts.txt ${prompts_sha:0:8}\n"

    echo -e "$summary"
    return $res
}

def get_lycoris_repo() {
    python -c """
import importlib.util
import pathlib
spec = importlib.util.find_spec('lycoris')
if spec is None:
    raise ImportError('lycoris module not found')
print(pathlib.Path(spec.origin).parent.parent)
    """
}

cleanup_empty_output() {
    [[ -n "$DEBUG" ]] && echo "\e[33mDEBUG\e[0m: Cleanup triggered for $OUTPUT_DIR/$NAME"
    [[ ! -d "$OUTPUT_DIR/$NAME" ]] && {
        [[ -n "$DEBUG" ]] && echo "\e[33mDEBUG\e[0m: Output directory doesn't exist, skipping cleanup"
        return 0
    }

    local samples=("$OUTPUT_DIR/$NAME"/**/*.{png}(N))
    local models=("$OUTPUT_DIR/$NAME"/**/*.{safetensors}(N))
    local git_repos=("$OUTPUT_DIR/$NAME"/**/.git(N))

    [[ -n "$DEBUG" ]] && {
        echo "\e[33mDEBUG\e[0m: Found ${#git_repos[@]} git repositories"
        echo "\e[33mDEBUG\e[0m: Found ${#samples[@]} sample files"
        echo "\e[33mDEBUG\e[0m: Found ${#models[@]} model files"
    }

    if [[ ${#samples[@]} -eq 0 && ${#models[@]} -eq 0 && ${#git_repos[@]} -eq 0 ]]; then
        if [[ -z "$NO_CLEAN" ]]; then
            echo "No samples or model files found, deleting empty output directory"
            rm -rf "$OUTPUT_DIR/$NAME"
        else
            echo "NO_CLEAN set, not deleting directory"
        fi
    else
        echo "Directory contains files, keeping it"
    fi
}
```

## What Gets Tracked

The script automatically tracks several key aspects of your training process. For Git repository states, it records the commit hashes of both the training script repository and the LyCORIS repository, allowing you to reference the exact code versions used.

The script also handles important training files by making copies of your configuration. It creates hashes of your training configuration file (`config.toml`), saves any sample prompts used during training in `sample-prompts.txt`, and preserves a copy of the training script itself for future reference.

To keep your workspace organized, the script includes automatic cleanup functionality. It monitors for any failed training runs and removes their output directories, ensuring your workspace stays clean and manageable over time.

## Helper Functions

The script relies on several helper functions:

### `setup_training_vars`

The `setup_training_vars` function handles the basic training variables needed for the process. It extracts both the dataset name and number of steps from the provided model name. Additionally, it creates and configures the necessary output directories while validating that the specified training directory exists.

### `setup_conda_env`

This function manages all aspects of the Conda environment setup. It handles activating the environment that was specified, validates that the environment actually exists, and performs the initialization of Conda for the current shell session.

### `store_commits_hashes`

The `store_commits_hashes` function is responsible for tracking the state of Git repositories. It records the commit hashes from the repositories, makes copies of all relevant configuration files, and generates SHA-1 hashes that can be used for tracking purposes.

### `cleanup_empty_output`

This cleanup function helps maintain a tidy workspace by removing output directories from failed training runs. It intelligently preserves any directories that contain samples or models while removing empty ones. For cases where this automatic cleanup is not desired, it can be disabled by setting `NO_CLEAN=1`.

## Best Practices

1. **Naming Convention**: Use a consistent naming format:

   ```bash
   {model}-{dataset}-v{version}s{steps}
   ```

   Example: `noob-surrounded_by_penis-v1s2400`

2. **Directory Structure**:

   ```bash
   datasets/
   ├── dataset_name/
   │   ├── config.toml
   │   └── sample-prompts.txt
   output_dir/
   └── model_name/
       ├── repos.git
       ├── config.toml
       ├── sample-prompts.txt
       └── training_script.sh
   ```

3. **Version Control**: Always work in Git repositories for:
   - Training scripts
   - Dataset configurations
   - Custom training code

4. **Documentation**: Keep notes of:
   - Training parameters that worked well
   - Failed experiments and why they failed
   - Model performance observations

You can enable debug output with:

```bash
DEBUG=1 ./your_training_script.sh
```

## Additional Tips

For effective version control, always work with training scripts, dataset configurations, and custom training code in a Git repository. Commit changes to your training scripts before starting new training runs to ensure reproducibility.

Keep detailed notes about your training process, including training parameters that worked well, failed experiments and their reasons, and model performance observations. This helps with future optimization and avoiding mistakes.

For long-term preservation, regularly backup your training configurations, sample prompts, and Git repositories. For efficient experiment tracking, consider using additional tools like TensorBoard for visualizing training metrics, Git LFS for large file storage, or external experiment tracking platforms for documenting the entire process.

---

<!--
HUGO_SEARCH_EXCLUDE_START
-->
{{< related-posts related="docs/yiff_toolkit/lora_training/ | docs/yiff_toolkit/lora_training/10-Minute-SDXL-LoRA-Training-for-the-Ultimate-Degenerates/ | docs/yiff_toolkit/" >}}
<!--
HUGO_SEARCH_EXCLUDE_END
-->
