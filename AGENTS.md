# Overview

This is a repo for conversion, storage, and inference of a nnUNet model format with a `safetensors` format and a unified `bundle.json` file. 

## Goal

The models should be completely independent of nnUNet, allowing for greater flexibility of input options (particular in production use cases). Specifically, we want to focus on the ability to do tensor in -> tensor out

## Tools

Use `uv` as the package manager of choice

Use `ruff` for formatting and checking

Use `mypy` for type checking
