---
myst:
    html_meta:
        description: "Creating a personal site with Jupyter Book is an interesting way to showcase your projects, research, or portfolio, especially if your work involves a lot of data analysis, coding, or scientific research."
        keywords: "Jupyter Book, Jupyter Notebooks, Sphinx, GitHub Pages, Netlify, Read the Docs, Hugo, Nikola, Jupyter ecosystem, MyST Markdown, Sphinx Book theme, GitHub Actions, automated deployments, Python, TensorFlow, kernels"
---

# Jupyter Book for a Personal Site

Having a personal site or blog was all the rage back in the early 2000s, but it seems to have fallen out of favor in recent years. Social media platforms like Twitter, Instagram, and LinkedIn have become the go-to places for sharing updates, thoughts, and projects. However, there's something special about having your own corner of the internet where you can showcase your work, share your ideas, and connect with others in a more long-form and structured way.

Creating a personal site with [Jupyter Book](https://jupyterbook.org/en/stable/intro.html) is an interesting way to showcase your projects, research, or portfolio, especially if your work involves a lot of data analysis, coding, or scientific research. Jupyter Book lets you build beautiful, publication-quality books and documents from computational material with relative ease. This whole site is built using it, and I've found it to be a great way to organize and share content.

It's a part of the whole Jupyter ecosystem, which includes [Jupyter Notebooks](https://jupyter.org), [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/index.html), and other tools for interactive computing, so if you're already familiar with those, you'll feel right at home with Jupyter Book. It's also built on top of [Sphinx](https://www.sphinx-doc.org/en/master/), which is a popular documentation tool in the Python community, so you get all the benefits of Sphinx as well. It supports a variety of content formats, including Markdown, Jupyter Notebooks, and reStructuredText, so you can mix and match to create the perfect book for your needs - it creates a static website that you can host anywhere, so you're not tied to any specific platform or service.

In my case, I host content on [GitHub Pages](https://pages.github.com), which is free and easy to set up. I can write content in Markdown or Jupyter Notebooks, push it to a GitHub repository, and have it automatically built and deployed. It's a simple and efficient workflow that lets me focus on creating content rather than worrying about the technical details of hosting and maintaining a website. You can also host your Jupyter Book on [Netlify](https://www.netlify.com), [Read the Docs](https://readthedocs.org) or just about anywhere which can host HTML content.

:::{tip}
When authoring content, if you use any Markdown, make sure you read the [JupyterBook MyST Markdown Guide](https://jupyterbook.org/content/myst.html), and the broader [MyST Markdown Guide](https://mystmd.org/guide) to understand how to use Markdown in Jupyter Book. It's a bit different from standard Markdown, but it offers an enourmous degree of authoring capabilities, at the cost of a bit of complexity and at times, perplexity!
:::

## Alternatives I tried

I previously had used [Hugo](https://gohugo.io) and [Nikola](https://getnikola.com), but neither of them felt quite right for my needs, nor did they integrate as well with my existing workflow. Jupyter Book just _feels_ like it works really well with a bunch of tools, and it's been a joy to use so far. Sure, it doesn't ship with all the templates and themes others do (pretty much there's one theme which works out of the box, [Sphinx Book](https://sphinx-book-theme.readthedocs.io/en/stable/), and which provides pretty much all you need), but it's way easier to setup and get going with without feeling as brittle or cumbersome.


## Automated deployments on GitHub Pages

A great feature of Jupyter Book is that it can automatically build and deploy your book to GitHub Pages whenever you push changes to your repository. This is done using GitHub Actions, which can fully automate your workflow in your GitHub repository. You can set up a simple workflow that builds your book whenever you push changes to your repository, and then deploys it to GitHub Pages so that it's always up to date.

 Here's my GitHub Actions workflow for deploying this site:

````{toggle}
```{code} yaml
name: deploy-book

# Run this when the master or main branch changes
on:
  push:
    branches:
    - master
    - main

jobs:
  deploy-book:
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    steps:
    - uses: actions/checkout@v3

    # Install dependencies
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11

    - name: Setup Graphviz
      # You may pin to the exact commit or the version.
      # uses: ts-graphviz/setup-graphviz@c001ccfb5aff62e28bda6a6c39b59a7e061be5b9
      uses: ts-graphviz/setup-graphviz@v1
      
    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    # Cache your executed notebooks between runs
    - name: cache executed notebooks
      uses: actions/cache@v3
      with:
        path: _build/.jupyter_cache
        key: jupyter-book-cache-${{ hashFiles('requirements.txt') }}

    # Build the book
    - name: Build the book
      run: |
        jupyter-book build .

    # Upload the book's HTML as an artifact
    - name: Upload artifact
      uses: actions/upload-pages-artifact@v2
      with:
        path: "_build/html"

    # Deploy the book's HTML to GitHub Pages
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v2
```
````

## Notes

Here are some peculiarities I've had to figure out the hard way:

- When using MyST Markdown content
  - Make sure you use the [correct syntax](https://jupyterbook.org/en/stable/reference/cheatsheet.html#code) for various blocks - for example, `{code-cell}` and `{code}` for executable code cells and code blocks, respectively.
  - Additionally, read the [text based MyST notebooks guide](https://myst-nb.readthedocs.io/en/latest/authoring/text-notebooks.html#) for MyST-NB, there is significant detail there which is useful to understand.
- Don't try and guess what your `_toc.yml` file should look like, you can programmatically generate it using the `jupyter-book toc` command. For example, in my case I just run `jupyter-book toc from-project -e .rst -e .md -e .ipynb -f jb-book pedros-experiments/ > pedros-experiments/_toc.yml` to generate the table of contents for this book.
- If you're using Jupyter Notebooks, make sure you're using the correct kernel for each notebook. You can specify the kernel in the notebook's metadata, or you can set a default kernel for the entire book in the `_config.yml` file. For example, in my case I use the `tf` kernel for most of my notebooks, which is a custom kernel I've set up for TensorFlow development, and I specify it in the notebook's metadata like so:
  ```yaml
  kernelspec:
    name: tf
    display_name: "Python (tf)"
    language: python
  ```
  To setup the kernel, use the following command: `python -m ipykernel install --user --name tf --display-name "Python (tf)"`.
