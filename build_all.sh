#!/bin/sh
# build.sh - Recursively build Quarto (.qmd or .ipynb) files in posts/,
# skipping any directory containing a .nobuild file.

build_file() {
    file="$1"
    echo "Building: $file"
    if ! quarto render "$file" --log build.log --log-level info; then
        echo "Error: Failed to build $file"
        exit 1
    fi
}

build_dir() {
    dir="$1"
    
    # Skip this directory if a .nobuild file is present.
    if [ -f "$dir/.nobuild" ]; then
        echo "Skipping directory: $dir (found .nobuild)"
        return
    fi

    for item in "$dir"/*; do
        [ -e "$item" ] || continue

        if [ -d "$item" ]; then
            build_dir "$item"
        elif [ -f "$item" ]; then
            case "$item" in
                *.qmd|*.ipynb)
                    build_file "$item"
                    ;;
            esac
        fi
    done
}

# Start the build process from the posts/ directory.
build_dir "posts"

# Build all else
build_dir "."
