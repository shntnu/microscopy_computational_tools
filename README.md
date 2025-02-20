# Microscopy computational tools

This repository contains scripts to identify cell centers and obtain cell embeddings with pretrained machine learning models. The scripts can be run locally or on the [Hail Batch service](https://hail.is/docs/batch/service.html).

The scripts are easy to modify, e.g., to add a new model.

## How to use

Start by using cellpose to identify cell centers. The output of the cellpose script can be used as input for the script that generates the embeddings.