---
file_format: mystnb
kernelspec:
    name: python3
    display_name: "Python (tf)"
    language: python
myst:
    html_meta:
        description: "A curated collection of experiments, how to's and general commentary, courtesy of the author."
        keywords: "Pedro, Data Science, Machine Learning, Software Engineering, Software Architecture, Python, Jupyter, JupyterBook, PyData-Sphinx"
---

# About $\Delta(\text{data})$

```{image} images/pedro-business.jpg
:alt: Pedro
:height: 350px
:align: right
:class: no-scaled-link
```

*$\Delta(\text{data})$* is a curated collection of experiments, [how to's](howtos/index) and general commentary, courtesy of the author. The best way to learn is to create, and a lot of what you will find here are the means by which I [experiment](experiments/index) and trial [thoughts](thoughts/index) and ideas. As such, readers are advised to approach these articles with a discerning eye and a generous pinch of skepticism.

Covering a spectrum of topics from software, machine learning, software architecture and perhaps, one or the other personal topic, what you find here may offer insights that range from interesting to mundane. However I hope that you find them useful, or at the very least, informative.

As the usual disclaimer, any views expressed here are my own and do not necessarily reflect the views of my employer, or any other entity with which I have been associated with.

```{code-cell} ipython3
:tags: [remove-input]
from datetime import datetime
from pathlib import Path
from typing import List
import re

def get_files(path: Path, ext: str, days: int = 7) -> List[Path]:
    return [f for f in path.rglob(f'*.{ext}') if (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < days]

def get_titles(files: List[Path]) -> List[str]:
    return [re.search(r'#(.*?)\\n', f.read_text()).group(1) for f in files]


def get_links(files: List[Path]):
    return [f'<a href="{f.with_suffix(".html")}">{t}</a>' for f, t in zip(files, get_titles(files))]

experiments = get_files(Path('experiments'), 'md') + get_files(Path('experiments'), 'ipynb')
thoughts = get_files(Path('thoughts'), 'md') + get_files(Path('thoughts'), 'ipynb')
howtos = get_files(Path('howtos'), 'md') + get_files(Path('howtos'), 'ipynb')

from IPython.display import HTML
experiments_links = '<br/>'.join(get_links(experiments[:2]))
thoughts_links = '<br/>'.join(get_links(thoughts[:2]))
howtos_links = '<br/>'.join(get_links(howtos[:2]))

if experiments_links or thoughts_links or howtos_links:
    display(HTML('<small><b>Recently Updated</b><br>'))
    # Display each if there are any, with a line break between each
    if experiments_links:
        display(HTML(f'{experiments_links}'))
    if thoughts_links:
        display(HTML(f'{thoughts_links}'))
    if howtos_links:
        display(HTML(f'{howtos_links}'))
    display(HTML('</small>'))
```

<small>Created with [JupyterBook](https://jupyterbook.org) and [PyData-Sphinx](https://pydata-sphinx-theme.readthedocs.io/en/stable/)</small>