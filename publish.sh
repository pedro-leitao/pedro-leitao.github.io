#!/bin/sh

touch docs/.nojekyll
git add .
git commit -m "Latest updates"
git push

