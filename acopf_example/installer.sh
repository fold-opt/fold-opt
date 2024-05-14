while read -r package; do
    conda install --name acopf "$package" || echo "Package $package not found, skipping..."
done < acopf/packages.txt
