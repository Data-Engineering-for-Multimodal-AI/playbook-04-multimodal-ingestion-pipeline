#!/bin/bash
# Setup script for ByteEats GitHub repository

# Set your GitHub username here
GITHUB_USERNAME="yourusername"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files to repository..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit of ByteEats Food Preparation ELT Pipeline"

# Setup GitHub remote (you'll need to create the repository on GitHub first)
echo "Setting up GitHub remote..."
git remote add origin https://github.com/$GITHUB_USERNAME/ByteEats.git

# Instructions for pushing
echo ""
echo "==================================================="
echo "Repository initialized successfully!"
echo "==================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create a new repository named 'ByteEats' on GitHub"
echo "   Visit: https://github.com/new"
echo ""
echo "2. Update the GITHUB_USERNAME in this script if needed"
echo ""
echo "3. Push your code to GitHub:"
echo "   git push -u origin main"
echo ""
echo "===================================================" 