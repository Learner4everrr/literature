@echo off
easyliter -i ./README.md -o ./Papers/

git add README.md
git commit -m update_readme
git push

pause