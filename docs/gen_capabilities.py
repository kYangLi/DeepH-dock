import re
from pathlib import Path
import json
import shutil


def clean_filename(filename):
    name = filename.replace('.ipynb', '')
    name = re.sub(r'[_-]+', ' ', name)
    name = name.title()
    return name

def get_notebook_title(notebook_path):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'markdown':
                content = ''.join(cell['source'])
                match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                if match:
                    return match.group(1).strip()
        return clean_filename(notebook_path.name)
    except:
        return clean_filename(notebook_path.name)

def generate_capability_rst(
    notebook_path: Path, rel_path: Path, capabilities_dir: Path
):
    notebook_name = notebook_path.stem
    title = get_notebook_title(notebook_path)
    #
    nb_relative_path = Path('..') / 'examples' / rel_path
    #
    target_dir = capabilities_dir / rel_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = capabilities_dir / rel_path
    #
    target_file.unlink(missing_ok=True)
    _tmp_path = Path('.')
    for _ in range(len(rel_path.parts)):
        _tmp_path /= '..'
    target_file.symlink_to(_tmp_path/nb_relative_path)
    print(f'Linked: {target_file}')
    #
    rst_relative_path = Path(rel_path).with_suffix('.ipynb')
    return str(rst_relative_path), title

def generate_capabilities_index(capabilities_dir, entries):
    content = """Capabilities
============

Welcome to the DeepH-dock capabilities library. It contains various use cases designed to help you get started quickly.

.. toctree::
    :maxdepth: 1
    :caption: Contents

"""
    #
    for entry_path, title in entries:
        module_list = [
            str(v).title().replace("_"," ") 
            for v in Path(entry_path).parts[:-2]
        ]
        module_str = "➯ ".join(module_list)
        content += f'    {module_str} ➯ {title} <{entry_path}>\n'
    #
    index_file = capabilities_dir / 'index.rst'
    index_file.write_text(content, encoding='utf-8')
    print(f'Generated: {index_file}')

def scan_examples_and_generate_capabilities():
    docs_dir = Path('.')
    examples_dir = docs_dir / '..' / 'examples'
    capabilities_dir = docs_dir / 'capabilities'
    
    if capabilities_dir.is_dir():
        shutil.rmtree(capabilities_dir)
    capabilities_dir.mkdir(exist_ok=False)
    
    entries = []
    
    for notebook_path in examples_dir.rglob('*.ipynb'):
        rel_path = notebook_path.relative_to(examples_dir)
        
        rst_relative_path, title = generate_capability_rst(
            notebook_path, rel_path, capabilities_dir
        )
        
        entries.append((rst_relative_path, title))
    
    if entries:
        generate_capabilities_index(capabilities_dir, entries)
        
        print(f'\nComplete! Dealing with {len(entries)} Notebooks.')
        print(f'Generated capabilities:')
        for rst_path, title in entries:
            print(f'  {rst_path} -> {title}')
    else:
        print('Connot find any .ipynb!')


def main():
    scan_examples_and_generate_capabilities()


if __name__ == '__main__':
    main()

