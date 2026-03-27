# Databuilder

This repository contains tools used to build SpecsQA dataset from raw, scraped html files.

Set up environment.
```bash
poetry install --with dev
```

You need to have the scraped data prepared for generating the dataset. You can use a copy of the data supplied with the repository, first unpack it:
```bash
cat ../scraped_data/scraped_data.tar.xz.* > scraped_data.tar.xz
tar -xvf scraped_data.tar.xz
```

The scraped data then needs to be parsed:
```bash
poetry run python -m databuilder.parse --input_dir scraped_data
```

This step will generate the JSON containing the dataframe with all the specs.

Next step is to perform preprocessing to the form expected by RAG system. Assuming that the data is stored in `scraped_data`, in order to run the preprocessing use the following command (where the preprocessed dataset will be stored in `dataset`):

```bash
poetry run python -m databuilder.dataset scraped_data dataset
```
