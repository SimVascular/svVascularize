#!/bin/bash
# Launcher script for svVascularize GUI.

# Add svVascularize to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

PLATFORM="$(uname -s)"
IS_LINUX=0
if [ "$PLATFORM" = "Linux" ]; then
    IS_LINUX=1
fi

# Function to find Mesa drivers
find_mesa_drivers() {
    local candidates=()

    # Check conda environment locations
    if [ -n "$CONDA_PREFIX" ]; then
        candidates+=(
            "$CONDA_PREFIX/lib/dri"
            "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/dri"
            "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib/dri"
        )

        # Check base conda if in an environment
        if [[ "$CONDA_PREFIX" == */envs/* ]]; then
            BASE_PREFIX="${CONDA_PREFIX%/envs/*}"
            candidates+=(
                "$BASE_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/dri"
            )
        fi
    fi

    # Check system locations (prioritize these as they exist)
    candidates=(
        "/usr/lib/x86_64-linux-gnu/dri"
        "/usr/lib64/dri"
        "/usr/lib/dri"
        "${candidates[@]}"
    )

    # Find first existing directory
    for dir in "${candidates[@]}"; do
        if [ -d "$dir" ] && [ -f "$dir/swrast_dri.so" -o -f "$dir/llvmpipe_dri.so" ]; then
            echo "$dir"
            return 0
        fi
    done

    return 1
}

# Parse command line arguments
USE_SYSTEM_GL=0
DEBUG_GL=0
for arg in "$@"; do
    case $arg in
        --system-gl)
            USE_SYSTEM_GL=1
            shift
            ;;
        --debug-gl)
            DEBUG_GL=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --system-gl    On Linux, use system OpenGL instead of software rendering"
            echo "  --debug-gl     Enable OpenGL debugging output"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SVV_GUI_GL_MODE='system'         On Linux, use system OpenGL (same as --system-gl)"
            echo "  SVV_GUI_SOFTWARE_DRIVER='swr'    On Linux, use OpenSWR instead of llvmpipe"
            echo "  SVV_LIBGL_DRIVERS_PATH='/path'   On Linux, override DRI drivers path"
            exit 0
            ;;
    esac
done

# Configure OpenGL settings
if [ "$IS_LINUX" -eq 1 ]; then
    if [ "$USE_SYSTEM_GL" -eq 1 ]; then
        export SVV_GUI_GL_MODE=system
        echo "Using system OpenGL drivers"
    else
        # Configure software rendering
        export LIBGL_ALWAYS_SOFTWARE=1
        export GALLIUM_DRIVER=llvmpipe
        export MESA_GL_VERSION_OVERRIDE=3.3
        export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
        export QT_OPENGL=software
        export SVV_GUI_GL_MODE=software

        # Find and set Mesa drivers path
        if [ -z "$SVV_LIBGL_DRIVERS_PATH" ]; then
            DRI_PATH=$(find_mesa_drivers)
            if [ $? -eq 0 ]; then
                export LIBGL_DRIVERS_PATH="$DRI_PATH"
                export SVV_LIBGL_DRIVERS_PATH="$DRI_PATH"
                echo "Found Mesa drivers at: $DRI_PATH"
            else
                echo "Warning: Could not find Mesa DRI drivers. Trying system fallback..."
                # Try installing mesa packages if needed
                if command -v conda &> /dev/null && [ -n "$CONDA_PREFIX" ]; then
                    echo "Attempting to install Mesa drivers via conda..."
                    conda install -y -c conda-forge mesa-libgl-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 2>/dev/null || true

                    # Try to find again after installation
                    DRI_PATH=$(find_mesa_drivers)
                    if [ $? -eq 0 ]; then
                        export LIBGL_DRIVERS_PATH="$DRI_PATH"
                        export SVV_LIBGL_DRIVERS_PATH="$DRI_PATH"
                        echo "Mesa drivers installed and found at: $DRI_PATH"
                    fi
                fi
            fi
        else
            export LIBGL_DRIVERS_PATH="$SVV_LIBGL_DRIVERS_PATH"
            echo "Using user-specified Mesa drivers at: $SVV_LIBGL_DRIVERS_PATH"
        fi

        # Force XCB on Wayland systems
        if [ -n "$WAYLAND_DISPLAY" ]; then
            export QT_QPA_PLATFORM=xcb
        fi
    fi
else
    if [ "$USE_SYSTEM_GL" -eq 1 ]; then
        echo "--system-gl is only relevant on Linux; using native platform rendering on $PLATFORM."
    fi
    echo "Using native platform rendering on $PLATFORM."
fi

# Enable debugging if requested
if [ "$DEBUG_GL" -eq 1 ]; then
    export SVV_GUI_DEBUG_GL=1
    if [ "$IS_LINUX" -eq 1 ]; then
        export MESA_DEBUG=1
        export LIBGL_DEBUG=verbose
    fi
fi

# Fix libffi conflicts for Mesa drivers
if [ "$IS_LINUX" -eq 1 ] && [ -n "$CONDA_PREFIX" ]; then
    # Look for libffi in conda environment
    if [ -f "$CONDA_PREFIX/lib/libffi.so.7" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.7:$LD_PRELOAD"
        echo "Using libffi from conda: $CONDA_PREFIX/lib/libffi.so.7"
    elif [ -f "$CONDA_PREFIX/lib/libffi.so.8" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libffi.so.8:$LD_PRELOAD"
        echo "Using libffi from conda: $CONDA_PREFIX/lib/libffi.so.8"
    fi
fi

# Launch the GUI (delegates to Python launcher which configures telemetry)
echo "Launching svVascularize GUI..."
python -c "
from svv.visualize.gui import launch_gui

launch_gui()
" || python -c "
from svv.visualize.gui import launch_gui

launch_gui()
"
