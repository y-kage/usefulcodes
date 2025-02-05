#!/bin/bash

dir_name=$(basename "$(pwd)")

# Remove existing index.html
rm -f index.html

# Start HTML structure
echo "<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">" > index.html
echo "<meta name="viewport" content="width=device-width, initial-scale=1.0">" >> index.html
echo "<title>$dir_name</title>" >> index.html
# echo "<style>table {width: 100%; border-collapse: collapse;} th, td {padding: 12px 20px; text-align: left;} th {font-weight: bold;}</style>" >> index.html
echo '<style>
table {width: 100%; border-collapse: collapse;}
th, td {padding: 12px 20px; text-align: left; vertical-align: middle;}
th {font-weight: bold; border-bottom: 2px solid #000;} /* Add line under the header row */
td:nth-child(2), th:nth-child(2) {text-align: right; width: 10%;} /* Align Size column */
td:nth-child(3), th:nth-child(3) {text-align: right; width: 20%;} /* Align Date Modified column */
hr {border: none; border-top: 2px solid #ccc; margin: 20px 0;}
</style>' >> index.html

echo "</head><body><h1>$dir_name</h1>" >> index.html

# Get list of unique extensions
extensions=$(for file in *; do
        if [ -f "$file" ] && [ "$file" != "index.html" ] && [ "$file" != "pub.sh" ]; then
                echo "${file##*.}"
        fi
done | sort | uniq)

echo '<table><tr><th>File Name</th><th>Size</th><th>Date Modified</th></tr></table>' >> index.html

# Loop through each extension group
for ext in $extensions; do
        echo "<table>" >> index.html

        for file in *."$ext"; do
                if [ -f "$file" ]; then
                        size=$(stat -c %s "$file" | numfmt --to=iec)
                        date=$(stat -c '%y' "$file" | awk '{print $1}' | xargs -I{} date -d {} "+%d %b %Y")
                        echo "<tr><td><a href=\"$file\" target=\"_blank\">$file</a></td><td>${size}</td><td>${date}</td></tr>" >> index.html
                fi
        done

        # Close table for this extension group
        echo '</table><hr>' >> index.html
done

# Close HTML structure
echo '</body></html>' >> index.html

# Set permissions for the generated index.html file
chmod 644 *
chmod +x pub.sh
