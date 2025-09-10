# TradeEngine 
  
## Routine to follow for every new session: 
 
0. Check if you are in the devel branch. If not, switch to the devel branch using:
   ```bash
   git checkout devel
   ```
1. Each tie after opening the project in pycharm, update the project from UI (or git pull) to get the latest changes. 
2. Open in-built terminal in pycharm and run the following command to snc the package dependencies:
   ```bash
   uv sync
   ``` 
3. After finishing working in the project, update the WH.md file with the hours worked. 
4. Commit the changes and push to the remote repository (devel - branch).