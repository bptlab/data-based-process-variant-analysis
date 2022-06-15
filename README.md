# Data-based Process Variant Analysis

This repository contains the python implementation of the Data-based Process Variant Analysis paper.

## Setup

To run the tool, you need to install a custom version of [ipyevents](https://github.com/mwcraig/ipyevents) (which is located in this repository as well).

Install the python requirements of this tool, like Jupyter Labs, PM4Py first. Then install the custom ipyevents package locally by 

```bash
$ cd custom-ipyevents
$ pip install -e .
$ jupyter nbextension install --py --symlink --sys-prefix ipyevents
$ jupyter nbextension enable --py --sys-prefix ipyevents
$ npm install
$ npm run build
$ jupyter labextension install
```

## Usage

To reproduce the findings of the paper, you can download the MIMIC-IV dataset on your own and generate an event log to feed into the tool. For doing so, you can use the Juypter notebook located in `notebooks/DBPVA_Event_Log_Generation.ipynb`.

With the generated event logs, or your own event logs, you can then use the `VariantComparator`.

The most simple way to use the tool is by only using the `VisualVariantComparator`. In `notebooks/Showcase.ipynb`there are some examples on how to use the package for variant comparison. 

To launch the tool, the following steps are required:

1. Read your event log
2. Split the event log by an arbitrary criterion
3. Initialize the Variant Comparator by 
```python
variant_comparator = VariantComparator(split_log_1, split_log_2, full_event_log, 'Name Split 1', 'Name Split 2')
variant_comparator.prepare()
```
4. Start the VisualVariantComparator by
```python
visual_variant_comparator = VisualVariantComparator(variant_comparator)
visual_variant_comparator.show()
```

If you only need specific parts of the variant comparator, have a look in the `vadbp/variant_comparator.py` file.