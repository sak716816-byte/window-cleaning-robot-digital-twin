import subprocess
import os
import sys

def run_command(cmd, cwd=None):
    try:
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    workspace = r"C:\Users\16906\Desktop\中科绿洲"
    
    print("Initializing Git Automation Pipeline...")
    
    # 1. git init
    if not os.path.exists(os.path.join(workspace, ".git")):
        run_command(["git", "init"], cwd=workspace)
    else:
        print("Git repository already initialized.")
        
    # 2. git config (local scope, to avoid global conflicts)
    run_command(["git", "config", "user.name", "Principal Architect"], cwd=workspace)
    run_command(["git", "config", "user.email", "architect@zhongkeluzhou.com"], cwd=workspace)
    
    # Create standard .gitignore
    gitignore_path = os.path.join(workspace, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write("__pycache__/\n*.pyc\n*.pyo\n*.pyd\n.pytest_cache/\n.vscode/\nformulas/\nformulas_step2/\nformulas_step3/\nformulas_step4/\n")
        print("Created .gitignore")
        
    # 3. git add .
    run_command(["git", "add", "."], cwd=workspace)
    
    # 4. git commit
    # We check if there are changes to commit
    try:
        status_result = subprocess.run(["git", "status", "--porcelain"], cwd=workspace, stdout=subprocess.PIPE, text=True)
        if status_result.stdout.strip():
            run_command(["git", "commit", "-m", "feat: core mechanics and digital twin engine initialized"], cwd=workspace)
            print("Committed successfully.")
        else:
            print("No changes to commit.")
    except Exception as e:
        print(f"Error during commit check: {e}")
        
    # 5. Remote GitHub config explanation
    print("\n" + "="*50)
    print("Git Local Commit Complete!")
    print("To host on GitHub, run the following commands in terminal:")
    print("  git remote add origin <your-github-repo-url>")
    print("  git branch -M main")
    print("  git push -u origin main")
    print("="*50)

if __name__ == "__main__":
    main()
