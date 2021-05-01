# PiSR-craft-beers

## Projects

### Scrapper

### Research

Researching models and data analysis.

#### Dependencies

Python 3.7

Create and activate virtual env. Then install `pip-tools`.

```sh
pip install pip-tools
```

To install dependencies run:

```sh
python -m piptools sync
```

When you did any changes to the `requirements.in`, please lock them before
pushing your changes.

```sh
python -m piptools compile --upgrade
```
