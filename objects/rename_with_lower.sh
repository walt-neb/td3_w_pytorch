for file in *.stl; do
    mv -- "$file" "$(tr '[:upper:]' '[:lower:]' <<< "${file:0:1}")${file:1}"
done
