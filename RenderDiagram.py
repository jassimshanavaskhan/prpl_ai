import subprocess
from pathlib import Path

def render_diagram(mermaid_content: str, output_file: str):
    """Render Mermaid diagram using CLI"""
    temp_file = Path('temp.mmd')
    try:
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Write content to temp file
        temp_file.write_text(mermaid_content)
        
        # Run mmdc command
        cmd = ['mmdc', '-i', str(temp_file), '-o', output_file,'-C', 'theme-neutral.json', '-b', 'white']
        subprocess.run(cmd, capture_output=True, text=True, shell=True, check=True)
        print(f"Diagram generated: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if temp_file.exists():
            temp_file.unlink()