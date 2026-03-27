# DualGraph

## Overview

This repository contains code for __DualGraph__ project.

## Installation

Install dependencies with:
```bash
poetry install --with dev
```

## Usage

### Step 1: Prepare data

In order to prepare data for indexing, follow instructions from `../databuilder` directory in this repository. This process should result in a directory named `dataset` with properly formatted data.

### Step 2: Prepare configuration file

See `configs/dumped_default_config.json` file. You can configure any field but you need to provide some mandatory fields, in particular: (1) address and key for remote LLM server (compatible with OpenAI API), (2) address and key for remote embedding server (compatible with OpenAI API), (3) address for remote RDFox server and (4) address for remote MLFlow server.

### Step 3: Indexing

Having the prepared dataset (`dataset`) and configuration file (`configuration.json`), you can run indexing process with:
```bash
poetry run python -m dualgraphrag.index \
    dataset/ \
    index/ \
    --config configuration.json
```
This will result in creation of index of the provided data in `index` directory.

### Step 4: Querying

To query the completed index, you need to provide questions in proper format (see `../questions/questions.json` in this repository). To run the querying, use:
```bash
poetry run python -m dualgraph.query \
    ../questions/questions.json \
    index/ \
    --run_name DualGraphQueryingRun \
    --config configuration.json
```
This will cause the file with results to appear in `index/query_results/`.

#### Settings

You can adjust querying mode via configuration file or by providing commandline arguments. For example, to use agentic querying, modify the configuration:
```json
{
  "general": {
    "use_which_system": "agentic"
  }
  "query": {
    "retrieval_methods": [
      "agentic"
    ]
  }
}
```
or add following arguments to the commandline:
```
    --extra general.use_which_system:\"agentic\" \
    --extra query.retrieval_methods:[\"agentic\"] 
```

### Step 5: Evaluation

For evaluation see `../evaluation` directory in this repository.

### Cache

For easier experimentation we support caching of results of LLM and embedding calls. This requires a running MongoDB docker container. Credentials are obtained from the environment variables (MONGO_PASSWORD, MONGO_USER, and MONGO_URL) - the variables should match the ones used to start the mongo server.

## Limitations

The solution can be used with various datasets, but importing the symbolic data is heavily customized per dataset. This customization is abstracted, so adapting to a different dataset should not be too hard.

### Symbolic import customization

Generally, each directory in the preprocessed dataset contains a json file describing its content. This config file contains a section "prescience", specifying which file contains the data, and what format the data is in. It is assumed that the knowledge is stored in a `.json` file. So, to implement your own format of data:
- store the data as `.json` or wrapped in `.json`
- pick a name for your format and make sure every file with knowledge to be injected is listed somewhere as prescience, with format set to this name
- add the name to the list of allowed values of the `Literal` in the definition of format field in `Prescience` class in `base.py`
- write a function to convert your data to triples (see below for more details) - of course that function may just be an entry point to a lot of complex code, that's up to you
- in `storage/rdfox/knowledge.py` add your format name to the converters dictionary, pointing it to the new function

The converter function should consume two arguments. The first one is whatever you loaded from the `.json`, the second one is a `NameCache` object you can use to preprocess IDs for the graph (you can do it your own way if you prefer). It should return three lists of strings:
- a list of triples in N3 format
- a list of datalog rules to be added to the dataset
- a list of IDs of entities appearing in the triples (used for alignment with information extracted from textual data)
Any of the above lists may of course be empty. Duplicates in the lists are not a problem apart from memory consumption.

The triples returned from your converter function will be upserted into RDFox.
