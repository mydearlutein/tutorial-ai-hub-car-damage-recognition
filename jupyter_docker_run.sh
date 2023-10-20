#!/bin/bash

docker run --network host -u jovyan -v `pwd`:/home/jovyan/work jupyter/base-notebook:python-3.9