#!/bin/bash

cd /app/docs

touch others.txt
cat others.txt > /tmp/others.txt

rm -rf /app/docs/*

# OCF Docs

mkdir -p /app/docs/ocf
if [ -d "/app/cache/ocf_mkdocs/.git" ]; then
    cd /app/cache/ocf_mkdocs
    git fetch origin
    git reset --hard origin/HEAD
    git clean -fd
    cd /app/docs
else
    git clone https://github.com/ocf/mkdocs /app/cache/ocf_mkdocs
fi

cp -a /app/cache/ocf_mkdocs/docs/* /app/docs/ocf/

# Dining

wget -O dining.html https://dining.berkeley.edu/menus/

# Joe

wget -O joe.html https://joewang.me/about

# Others

cp /tmp/others.txt others.txt

# Done

echo "✅ Sync complete!"
